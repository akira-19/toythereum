mod compiler;
mod parser;

fn main() {
    let res = compiler::run();
    match res {
        Ok(_) => println!("Compilation successful"),
        Err(e) => println!("Compilation failed: {}", e),
    }
}
