use rlp::RlpStream;
use tiny_keccak::{Hasher, Keccak};

use crate::state::Address;

pub fn keccak256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Keccak::v256();
    let mut output = [0u8; 32];
    hasher.update(data);
    hasher.finalize(&mut output);
    output.to_vec()
}

pub fn hex_to_vec(hex: &str) -> Result<Vec<u8>, String> {
    let hex_string;
    if hex.starts_with("0x") {
        hex_string = &hex[2..];
    } else {
        hex_string = hex;
    }
    (0..hex_string.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

pub fn vec_to_hex(vec: Vec<u8>) -> String {
    let mut hex_string = String::new();
    hex_string.push_str("0x");
    for byte in vec {
        hex_string.push_str(&format!("{:02x}", byte));
    }
    hex_string
}

pub fn calculate_contract_address(sender_address: Address, nonce: u64) -> Address {
    // address to bytes
    let address = hex_to_vec(sender_address.as_str()).unwrap();
    // rlp encode
    let mut rlp_stream = RlpStream::new();
    rlp_stream.begin_list(2); // リストの開始
    rlp_stream.append(&address); // アドレスの追加
    rlp_stream.append(&nonce); // ナンスの追加
    let rlp_encoded = rlp_stream.out();

    // keccak256
    let mut hasher = Keccak::v256();
    let mut hash_output = [0u8; 32];
    hasher.update(&rlp_encoded);
    hasher.finalize(&mut hash_output);

    // last 20 bytes
    let mut contract_address = [0u8; 20];
    contract_address.copy_from_slice(&hash_output[12..32]);

    Address::new(vec_to_hex(contract_address.to_vec()).as_str())
}
