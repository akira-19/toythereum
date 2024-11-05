use crate::state::*;
use primitive_types::U256;

pub struct EVM {
    stack: Stack,
    memory: Vec<u8>,
    pc: usize,
    pub gas: U256,
    returns: Option<ReturnValue>,
    code: Vec<u8>,
    ee: ExecutionEnvironment,
}

pub struct Input {
    contract_address: Address,
    sender: Address,
    gas_price: U256,
    calldata: Vec<u8>,
    value: U256,
}

pub struct ExecutionEnvironment {
    input: Input,
    gas: U256,
    state: WorldState,
    code: CodeStorage,
}

impl ExecutionEnvironment {
    pub fn new(
        contract_address: Address,
        sender: Address,
        gas_price: U256,
        calldata: Vec<u8>,
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
                calldata: calldata,
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

#[derive(Clone)]
pub enum ReturnValue {
    Return(Vec<u8>),
    Revert(Vec<u8>),
    Stop,
}

impl EVM {
    pub fn new(
        contract_address: Address,
        sender: Address,
        gas_price: U256,
        calldata: Vec<u8>,
        value: U256,
        gas: U256,
        state: WorldState,
        code: CodeStorage,
    ) -> EVM {
        EVM {
            stack: Stack::new(),
            memory: vec![0u8; 128],
            pc: 0,
            gas,
            returns: None,
            code: Vec::new(),
            ee: ExecutionEnvironment::new(
                contract_address,
                sender,
                gas_price,
                calldata,
                value,
                gas,
                state,
                code,
            ),
        }
    }

    pub fn get_returns(&self) -> Option<ReturnValue> {
        self.returns.clone()
    }

    fn consume_gas(&mut self, gas: U256) {
        let (res, underflow) = self.gas.overflowing_sub(gas);
        if underflow {
            self.returns = Some(ReturnValue::Revert(Vec::new()));
        } else {
            self.gas = res;
        }
    }

    pub fn run(&mut self, mut storage: &mut StorageTrie) {
        let contract_address = self.ee.input.contract_address.clone();
        let initial_node = storage.get(&contract_address).clone();

        self.code = self.ee.code.get_code(&contract_address).unwrap().to_vec();
        loop {
            println!("stack: {:?}", self.stack.data);
            let opcode = self.code[self.pc];
            println!("opcode: {:02X}", opcode);
            match opcode {
                0x00 => self.op_stop(),
                0x01 => self.op_add(),
                0x02 => self.op_mul(),

                0x03 => self.op_sub(),
                0x04 => self.op_div(),

                0x10 => self.op_lt(),

                0x14 => self.op_eq(),

                0x1C => self.op_shr(),

                0x35 => self.op_calldataload(),

                0x51 => self.op_mload(),
                0x52 => self.op_mstore(),
                0x54 => self.op_sload(storage),
                0x55 => self.op_sstore(storage),

                0x56 => self.op_jump(),
                0x57 => self.op_jumpi(),
                0x5b => self.op_jumpdest(),

                0x60 => self.op_push(1),
                0x7F => self.op_push(32),
                0xF3 => self.op_return(),
                0xFD => self.op_revert(),
                _ => panic!("Invalid opcode"),
            }

            self.pc += 1;

            if self.pc >= self.code.len() {
                break;
            }

            match self.returns {
                Some(ReturnValue::Return(_)) => break,
                Some(ReturnValue::Revert(_)) => {
                    storage.rollback(&contract_address, initial_node);
                    break;
                }
                Some(ReturnValue::Stop) => break,
                None => continue,
            }
        }
    }

    fn op_stop(&mut self) {
        // stop the execution
        self.returns = Some(ReturnValue::Stop);
    }

    fn op_add(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        self.stack.push(a + b);
        self.consume_gas(U256::from(3));
    }

    fn op_mul(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        self.stack.push(a * b);
        self.consume_gas(U256::from(5));
    }

    fn op_sub(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        self.stack.push(a - b);
        self.consume_gas(U256::from(3));
    }

    fn op_div(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        self.stack.push(a / b);
        self.consume_gas(U256::from(5));
    }

    fn op_lt(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        if a < b {
            self.stack.push(U256::one());
        } else {
            self.stack.push(U256::zero());
        }
        self.consume_gas(U256::from(3));
    }

    fn op_eq(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        if a == b {
            self.stack.push(U256::one());
        } else {
            self.stack.push(U256::zero());
        }
        self.consume_gas(U256::from(3));
    }

    fn op_shr(&mut self) {
        let a = self.stack.pop();
        let b = self.stack.pop();
        println!("a: {}, b: {}", a, b);
        self.stack.push(a >> b);
        self.consume_gas(U256::from(3));
    }

    fn op_calldataload(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let value = U256::from_big_endian(&self.ee.input.calldata[offset..offset + 32]);
        self.stack.push(value);
        self.consume_gas(U256::from(3));
    }

    fn op_mload(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let value = U256::from_big_endian(&self.memory[offset..offset + 32]);
        self.stack.push(value);
        self.consume_gas(U256::from(3));
    }

    fn op_mstore(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let value = self.stack.pop();
        let value_bytes = value.to_big_endian();
        for (i, byte) in value_bytes.iter().enumerate() {
            self.memory.insert(offset + i, *byte);
        }
        self.consume_gas(U256::from(3));
    }

    fn op_sload(&mut self, storage: &StorageTrie) {
        let key = self.stack.pop();
        let value = storage.get_value(&self.ee.input.contract_address, key);
        self.stack.push(value);
        self.consume_gas(U256::from(100));
    }

    fn op_sstore(&mut self, storage: &mut StorageTrie) {
        let key = self.stack.pop();
        let value = self.stack.pop();
        let address = self.ee.input.contract_address.clone();
        storage.upsert(address, key, value);
        self.consume_gas(U256::from(100));
    }

    fn op_jump(&mut self) {
        let position = self.stack.pop().as_usize();
        self.pc = position;
        self.consume_gas(U256::from(8));
    }

    fn op_jumpi(&mut self) {
        let position = self.stack.pop().as_usize();
        let condition = self.stack.pop();
        if condition != U256::zero() {
            self.pc = position;
        }
        self.consume_gas(U256::from(10));
    }

    fn op_jumpdest(&mut self) {
        self.consume_gas(U256::from(1));
    }

    fn op_push0(&mut self) {
        self.stack.push(U256::from(0));
        self.pc += 1;
        self.consume_gas(U256::from(2));
    }

    fn op_push(&mut self, length: usize) {
        let value = U256::from_big_endian(&self.code[self.pc + 1..self.pc + 1 + length]);

        self.pc += length;
        self.stack.push(value);
        self.consume_gas(U256::from(3));
    }

    fn op_return(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;

        let return_value = &self.memory[offset..offset + length];
        self.returns = Some(ReturnValue::Return(Vec::from(return_value)));
    }

    fn op_revert(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;

        let return_value = &self.memory[offset..offset + length];
        self.returns = Some(ReturnValue::Revert(Vec::from(return_value)));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::state;

    use super::*;

    #[test]
    fn test_addition() {
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
        let mut evm = EVM::new(
            contract_address,
            Address::new("0x1234567890123456789012345678901234567890"),
            U256::zero(),
            Vec::new(),
            U256::zero(),
            U256::from(1000),
            WorldState::new(),
            code_storage,
        );

        let mut storage = state::StorageTrie(HashMap::new());

        evm.run(&mut storage);

        // assert_eq!(evm.returns, Some(ReturnValue::Return([3u8])));
    }
}
