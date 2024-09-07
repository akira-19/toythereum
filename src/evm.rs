use crate::state::*;
use primitive_types::U256;

struct EVM {
    stack: Stack,
    memory: Vec<u8>,
    pc: usize,
    gas: U256,
    returns: Vec<u8>,
    code: Vec<u8>,
}

struct Input {
    contract_address: Address,
    sender: Address,
    gas_price: U256,
    data: Vec<u8>,
    value: U256,
}

struct ExecutionEnvironment {
    input: Input,
    gas: U256,
    state: WorldState,
    code: CodeStorage,
}

impl ExecutionEnvironment {
    fn new(
        contract_address: Address,
        sender: Address,
        gas_price: U256,
        data: Vec<u8>,
        value: U256,
        gas: U256,
        state: WorldState,
        code: CodeStorage,
    ) -> ExecutionEnvironment {
        ExecutionEnvironment {
            input: Input {
                contract_address: contract_address,
                sender: sender,
                gas_price: gas_price,
                data: data,
                value: value,
            },
            gas: gas,
            state: state,
            code: code,
        }
    }
}

struct Stack {
    data: Vec<U256>,
    sp: usize,
}

impl Stack {
    fn new() -> Stack {
        Stack {
            data: Vec::new(),
            sp: 0,
        }
    }

    fn push(&mut self, value: U256) {
        self.data.push(value);
        self.sp += 1;
    }

    fn pop(&mut self) -> U256 {
        let v = self.data.pop().unwrap();
        self.sp -= 1;
        v
    }
}

impl EVM {
    fn new() -> EVM {
        EVM {
            stack: Stack::new(),
            memory: Vec::new(),
            pc: 0,
            gas: U256::zero(),
            returns: Vec::new(),
            code: Vec::new(),
        }
    }

    fn run(&mut self, ee: ExecutionEnvironment) {
        let contract_address = ee.input.contract_address;
        self.code = ee.code.get_code(&contract_address).unwrap().to_vec();
        loop {
            let opcode = self.code[self.pc];
            println!("opcode: {:?}", opcode);
            match opcode {
                0x00 => self.op_stop(),
                0x01 => self.op_add(),
                0x52 => self.op_mstore(),
                0x60 => self.op_push(1),
                0xF3 => self.op_return(),
                _ => panic!("Invalid opcode"),
            }

            self.pc += 1;

            if self.pc >= self.code.len() {
                break;
            }
        }
    }

    fn op_stop(&self) {
        // stop the execution
    }

    fn op_add(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        self.stack.push(a + b);
    }

    fn op_mstore(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let value = self.stack.pop();
        let mut value_bytes = [0u8; 32];
        value.to_big_endian(&mut value_bytes);
        for (i, byte) in value_bytes.iter().enumerate() {
            self.memory.insert(offset + i, *byte);
        }
    }

    fn op_push0(&mut self) {
        self.stack.push(U256::from(0));
        self.pc += 1;
    }

    fn op_push(&mut self, length: usize) {
        let value = U256::from_big_endian(&self.code[self.pc + 1..self.pc + 1 + length]);

        self.pc += length;
        // self.gas -= U256::from(3);
        self.stack.push(value);
    }

    fn op_return(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;

        let return_value = &self.memory[offset..offset + length];
        self.returns = Vec::from(return_value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let mut evm = EVM::new();
        let code = vec![
            0x60, 0x01, // PUSH1
            0x60, 0x02, // PUSH1
            0x01, // ADD
            0x60, 0x00, // PUSH1
            0x52, // MSTORE
            0x60, 0x20, // PUSH1
            0x60, 0x00, // PUSH1
            0xF3, // RETURN
        ];
        // [74] PUSH1 0x02
        // [75] PUSH1 0x01
        // [76] ADD
        // [77] DUP1
        // [78] PUSH1 0x40
        // [79] MSTORE
        // [80] PUSH1 0x20
        // [81] PUSH1 0x40
        // [82] RETURN

        // PUSH1 0x01    // スタックに1をプッシュ
        // PUSH1 0x02    // スタックに2をプッシュ
        // ADD           // スタックの値を足して結果をスタックにプッシュ
        // PUSH1 0x00    // メモリの開始位置（オフセット0）
        // MSTORE        // メモリにスタックの値を保存
        // PUSH1 0x20    // 返すデータのサイズ（32バイト）
        // PUSH1 0x00    // メモリの開始位置（オフセット0）
        // RETURN
        let contract_address = Address::new("0x1234567890123456789012345678901234567890");
        let mut code_storage = CodeStorage::new();
        code_storage.insert_code(contract_address.clone(), code.clone());
        let ee = ExecutionEnvironment::new(
            contract_address,
            Address::new("0x1234567890123456789012345678901234567890"),
            U256::zero(),
            Vec::new(),
            U256::zero(),
            U256::from(1000),
            WorldState::new(),
            code_storage,
        );

        evm.run(ee);

        assert_eq!(evm.returns[evm.returns.len() - 1], 3u8);
    }
}
