use std::collections::HashMap;

use crate::{
    state::*,
    util::{calculate_contract_address, keccak256},
};
use primitive_types::U256;

pub struct EVM {
    stack: Stack,
    memory: Vec<u8>,
    pc: usize,
    pub gas: U256,
    returns: Option<ReturnValue>,
    code: Vec<u8>,
    ee: ExecutionEnvironment,
    pub contract_address: Option<Address>,
}

pub struct Input {
    from: Address,
    to: Option<Address>,
    gas_price: U256,
    calldata: Vec<u8>,
    value: U256,
}

pub struct ExecutionEnvironment {
    input: Input,
    gas: U256,
}

impl ExecutionEnvironment {
    pub fn new(
        from: Address,
        to: Option<Address>,
        gas_price: U256,
        calldata: Vec<u8>,
        value: U256,
        gas: U256,
    ) -> ExecutionEnvironment {
        ExecutionEnvironment {
            input: Input {
                from,
                to,
                gas_price: gas_price,
                calldata: calldata,
                value: value,
            },
            gas: gas,
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
        from: Address,
        to: Option<Address>,
        gas_price: U256,
        calldata: Vec<u8>,
        value: U256,
        gas: U256,
    ) -> EVM {
        EVM {
            stack: Stack::new(),
            memory: vec![0u8; 128],
            pc: 0,
            gas,
            returns: None,
            code: Vec::new(),
            ee: ExecutionEnvironment::new(from.clone(), to, gas_price, calldata, value, gas),
            contract_address: None,
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

    pub fn run(
        &mut self,
        storage: &mut StorageTrie,
        code_storage: &mut CodeStorage,
        state: &mut WorldState,
    ) {
        let to = self.ee.input.to.clone();

        let contract_address = if let Some(contract) = to.clone() {
            contract
        } else {
            let from = self.ee.input.from.clone();
            let state_clone = state.clone();
            let account = state_clone.get_account(&from.clone()).unwrap();
            calculate_contract_address(from, U256::low_u64(&account.nonce))
        };
        self.contract_address = Some(contract_address.clone());

        let initial_node;

        if let Some(contract) = to {
            self.code = code_storage.get_code(&contract).unwrap().to_vec();
            initial_node = storage.get(&contract).clone();
            self.code = code_storage.get_code(&contract).unwrap().to_vec();
        } else {
            initial_node = HashMap::new();
            self.code = self.ee.input.calldata.clone();
            let new_account = AccountState::new();
            state.upsert_account(self.contract_address.clone().unwrap(), new_account);
        }

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

                0x39 => self.op_codecopy(),

                0x51 => self.op_mload(),
                0x52 => self.op_mstore(),
                0x54 => self.op_sload(storage),
                0x55 => self.op_sstore(storage),

                0x56 => self.op_jump(),
                0x57 => self.op_jumpi(),
                0x5b => self.op_jumpdest(),

                0x5F => self.op_push0(),
                0x60 => self.op_push(1),
                0x7F => self.op_push(32),

                0x80 => self.op_dup(1),

                0xF3 => self.op_return(code_storage, state),
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
                    storage.rollback(&self.contract_address.clone().unwrap(), initial_node);
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

    fn op_codecopy(&mut self) {
        let dest_offset = self.stack.pop().as_u32() as usize;
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;
        // let code = code_storage.get_code(&self.contract_address).unwrap();
        let code = self.ee.input.calldata.clone();
        let code_slice = &code[offset..offset + length];

        let current_memory_size = ((self.memory.len() + 31) / 32) as u32;

        // TODO: need to check if the code size is longer than the current memory size
        for (i, byte) in code_slice.iter().enumerate() {
            self.memory.insert(dest_offset + i, *byte);
        }

        let new_memory_size = ((self.memory.len() + 31) / 32) as u32;

        // TODO: need to check the gas cost calculation
        let memory_expansion_cost = (new_memory_size.pow(2) - current_memory_size.pow(2)) / 512
            + 3 * (new_memory_size - current_memory_size);

        let gas_cost = 3 + 3 * new_memory_size + memory_expansion_cost;

        self.consume_gas(U256::from(gas_cost));
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
        // TODO: gas should be calculated based on the memory expansion
        self.consume_gas(U256::from(3));
    }

    fn op_sload(&mut self, storage: &StorageTrie) {
        let key = self.stack.pop();
        let value = storage.get_value(&self.contract_address.clone().unwrap(), key);
        self.stack.push(value);
        self.consume_gas(U256::from(100));
    }

    fn op_sstore(&mut self, storage: &mut StorageTrie) {
        let key = self.stack.pop();
        let value = self.stack.pop();
        let address = self.contract_address.clone().unwrap();
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
        self.consume_gas(U256::from(2));
    }

    fn op_push(&mut self, length: usize) {
        let value = U256::from_big_endian(&self.code[self.pc + 1..self.pc + 1 + length]);

        self.pc += length;
        self.stack.push(value);
        self.consume_gas(U256::from(3));
    }

    fn op_dup(&mut self, n: usize) {
        let value = self.stack.data[self.stack.sp - n];
        self.stack.push(value);
        self.consume_gas(U256::from(3));
    }

    fn op_return(&mut self, code_storage: &mut CodeStorage, state: &mut WorldState) {
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;

        let return_value = &self.memory[offset..offset + length];

        if self.ee.input.to == None {
            let account = state
                .get_mut_account(&self.contract_address.clone().unwrap())
                .unwrap();
            account.code_hash = keccak256(&return_value.to_vec());
            code_storage.insert_code(
                self.contract_address.clone().unwrap(),
                return_value.to_vec(),
            );
        }

        self.returns = Some(ReturnValue::Return(Vec::from(return_value)));
    }

    fn op_revert(&mut self) {
        let offset = self.stack.pop().as_u32() as usize;
        let length = self.stack.pop().as_u32() as usize;

        let return_value = &self.memory[offset..offset + length];
        self.returns = Some(ReturnValue::Revert(Vec::from(return_value)));
    }
}
