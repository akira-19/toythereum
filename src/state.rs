use primitive_types::U256;
use regex::Regex;
use std::collections::HashMap;

pub struct WorldState {
    accounts: HashMap<Address, AccountState>,
}

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            accounts: HashMap::new(),
        }
    }

    fn get_account(&self, address: &Address) -> Option<&AccountState> {
        self.accounts.get(address)
    }

    fn insert_account(&mut self, address: Address, account: AccountState) {
        self.accounts.insert(address, account);
    }
}

#[derive(Hash, Eq, PartialEq, Debug)]
struct AccountState {
    nonce: U256,
    balance: U256,
    storage_root: U256, // merkle root of the storage trie
    code_hash: Vec<u8>,
}

pub struct CodeStorage {
    code: HashMap<Address, Vec<u8>>,
}

impl CodeStorage {
    pub fn new() -> CodeStorage {
        CodeStorage {
            code: HashMap::new(),
        }
    }

    pub fn get_code(&self, address: &Address) -> Option<&Vec<u8>> {
        self.code.get(address)
    }

    pub fn insert_code(&mut self, address: Address, code: Vec<u8>) {
        self.code.insert(address, code);
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct Address(String);

impl Address {
    pub fn new(s: &str) -> Address {
        let re = Regex::new(r"0x[0-9a-fA-F]{40}").unwrap();

        if !re.is_match(s) {
            panic!("Invalid address format");
        }
        Address(s.to_string())
    }
}

impl AccountState {
    fn new() -> AccountState {
        AccountState {
            nonce: U256::zero(),
            balance: U256::zero(),
            storage_root: U256::zero(),
            code_hash: Vec::new(),
        }
    }
}

type StorageTrie = HashMap<U256, U256>;