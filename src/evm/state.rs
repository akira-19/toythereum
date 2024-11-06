use primitive_types::U256;
use regex::Regex;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct WorldState {
    accounts: HashMap<Address, AccountState>,
}

impl WorldState {
    pub fn new() -> WorldState {
        WorldState {
            accounts: HashMap::new(),
        }
    }

    pub fn get_account(&self, address: &Address) -> Option<&AccountState> {
        self.accounts.get(address)
    }

    pub fn insert_account(&mut self, address: Address, account: AccountState) {
        self.accounts.insert(address, account);
    }
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct AccountState {
    pub nonce: U256,
    balance: U256,
    storage_root: U256, // merkle root of the storage trie
    pub code_hash: Vec<u8>,
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

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AccountState {
    pub fn new() -> AccountState {
        AccountState {
            nonce: U256::zero(),
            balance: U256::zero(),
            storage_root: U256::zero(),
            code_hash: Vec::new(),
        }
    }
}

pub type StorageTrieNode = HashMap<U256, U256>;

#[derive(Debug, Clone)]

pub struct StorageTrie(pub HashMap<Address, StorageTrieNode>);

impl StorageTrie {
    pub fn new() -> StorageTrie {
        StorageTrie(HashMap::new())
    }

    pub fn get(&self, address: &Address) -> StorageTrieNode {
        if let Some(node) = self.0.get(address) {
            node.clone()
        } else {
            HashMap::new()
        }
    }

    pub fn get_value(&self, address: &Address, key: U256) -> U256 {
        self.0.get(address).unwrap().get(&key).cloned().unwrap()
    }

    pub fn rollback(&mut self, address: &Address, node: StorageTrieNode) {
        let current_node = self.0.get_mut(address).unwrap();
        *current_node = node;
    }

    pub fn upsert(&mut self, address: Address, key: U256, value: U256) {
        if let Some(node) = self.0.get_mut(&address) {
            node.insert(key, value);
        } else {
            let mut node = HashMap::new();
            node.insert(key, value);
            self.0.insert(address, node);
        }
    }
}
