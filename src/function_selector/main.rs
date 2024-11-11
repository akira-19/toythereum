use std::env;

use tiny_keccak::{Hasher, Keccak};

fn calculate_function_selector(signature: &str) -> String {
    let mut hasher = Keccak::v256();
    let mut output = [0u8; 32];

    // ハッシュに関数シグネチャを入力
    hasher.update(signature.as_bytes());
    hasher.finalize(&mut output);

    // 先頭4バイトをfunction selectorとして返す
    format!(
        "{:02x}{:02x}{:02x}{:02x}",
        output[0], output[1], output[2], output[3]
    )
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <function_signature>", args[0]);
        return;
    }

    let signature = &args[1];

    let selector = calculate_function_selector(signature);

    println!("Function selector: {:?}", selector);
}
