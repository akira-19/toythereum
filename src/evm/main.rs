mod evm;
mod state;
mod util;
use primitive_types::U256;
use serde::{Deserialize, Serialize};
use std::io::{self, BufReader};
use std::{collections::HashMap, fs::File};
use util::{hex_to_vec, vec_to_hex};

#[derive(Debug, Serialize, Deserialize)]

struct TxInput {
    from: state::Address,
    to: Option<state::Address>,
    value: U256,
    data: String,
    gas_limit: U256,
    gas_price: U256,
    nonce: U256,
}

fn init_state() -> state::WorldState {
    let mut ws = state::WorldState::new();
    let account = state::AccountState::new();
    ws.upsert_account(
        state::Address::new("0x1234567890abcdef1234567890abcdef12345678"),
        account,
    );
    ws
}

// fn receive_transaction(input: TxInput) -> evm::EVM {
//     evm::EVM::new(
//         input.from,
//         input.to,
//         input.value,
//         input.data,
//         input.gas_limit,
//         input.gas_price,
//         ws,
//     )
// }

fn main() {
    let mut ws = init_state();
    let mut code_storage = state::CodeStorage::new();
    let mut storage = state::StorageTrie(HashMap::new());

    // let mut calldata = 3405006467u32.to_be_bytes().to_vec();
    // let attr = U256::from(11).to_big_endian().to_vec();
    // calldata.extend(attr);

    let mut line = String::new();
    while io::stdin().read_line(&mut line).unwrap() > 0 {
        // read json
        if line.ends_with(".json") {
            let file = File::open(line.clone()).unwrap();
            let reader = BufReader::new(file);

            let input: TxInput = serde_json::from_reader(reader).unwrap();

            let mut evm = evm::EVM::new(
                input.from,
                input.to,
                input.gas_price,
                hex_to_vec(input.data.as_str()).unwrap(),
                input.value,
                input.gas_limit,
                ws.clone(),
            );

            evm.run(&mut storage, &mut code_storage);

            if let Some(ret) = evm.get_returns() {
                match ret {
                    evm::ReturnValue::Revert(v) => {
                        println!("Revert: {:?}", v);
                    }
                    evm::ReturnValue::Return(v) => {
                        println!("Return: {:?}", v);
                    }
                    evm::ReturnValue::Stop => {
                        println!("Stop");
                    }
                }
            } else {
                println!("No return value");
            }
        }
        line.clear();
    }

    // output code storage
    for (k, v) in code_storage.code.iter() {
        println!("CodeStorage: {:?} {:?}", k, vec_to_hex(v.clone()));
    }
}
