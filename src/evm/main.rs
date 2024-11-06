mod evm;
mod state;
mod util;
use primitive_types::U256;
use std::collections::HashMap;

fn main() {
    let mut ws = state::WorldState::new();
    let mut account = state::AccountState::new();
    account.code_hash =
        hex_to_vec("047988cc6504a844244fa59491ee56b4a94e022becfee4653b5f685a76816f36").unwrap();
    ws.insert_account(
        state::Address::new("0x1234567890abcdef1234567890abcdef12345678"),
        account,
    );
    let mut code = state::CodeStorage::new();
    code.insert_code(
        state::Address::new("0x1234567890abcdef1234567890abcdef12345678"),
        hex_to_vec("7f00000000000000000000000000000000000000000000000000000000000000807f0000000000000000000000000000000000000000000000000000000000000040527f00000000000000000000000000000000000000000000000000000000000000047f00000000000000000000000000000000000000000000000000000000000000037f00000000000000000000000000000000000000000000000000000000000000027f00000000000000000000000000000000000000000000000000000000000000017f000000000000000000000000000000000000000000000000000000000000000b7f0000000000000000000000000000000000000000000000000000000000000000557f00000000000000000000000000000000000000000000000000000000000000e07f0000000000000000000000000000000000000000000000000000000000000000351c7f00000000000000000000000000000000000000000000000000000000f8a8fd6d147f0000000000000000000000000000000000000000000000000000000000000307577f00000000000000000000000000000000000000000000000000000000000000e07f0000000000000000000000000000000000000000000000000000000000000000351c7f00000000000000000000000000000000000000000000000000000000caf44683147f000000000000000000000000000000000000000000000000000000000000025d577f00000000000000000000000000000000000000000000000000000000000000007f0000000000000000000000000000000000000000000000000000000000000000fd5b7f0000000000000000000000000000000000000000000000000000000000000004357f000000000000000000000000000000000000000000000000000000000000000054107f0000000000000000000000000000000000000000000000000000000000000080527f00000000000000000000000000000000000000000000000000000000000000207f0000000000000000000000000000000000000000000000000000000000000080f35b7f0000000000000000000000000000000000000000000000000000000000000000547f0000000000000000000000000000000000000000000000000000000000000002017f0000000000000000000000000000000000000000000000000000000000000080527f0000000000000000000000000000000000000000000000000000000000000080517f00000000000000000000000000000000000000000000000000000000000000a0527f00000000000000000000000000000000000000000000000000000000000000207f00000000000000000000000000000000000000000000000000000000000000a0f3").unwrap()
    );

    // test1: 4171824493
    // test2: 3405006467

    let mut calldata = 3405006467u32.to_be_bytes().to_vec();
    let attr = U256::from(11).to_big_endian().to_vec();
    calldata.extend(attr);

    let mut evm = evm::EVM::new(
        state::Address::new("0x1234567890abcdef1234567890abcdef12345678"),
        state::Address::new("0x1234567890abcdef1234567890abcdef12345678"),
        U256::zero(),
        calldata,
        U256::zero(),
        U256::from(10000),
        ws,
        code,
    );

    let mut storage = state::StorageTrie(HashMap::new());

    evm.run(&mut storage);

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

    println!("Gas: {}", evm.gas);
}
