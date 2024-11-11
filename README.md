# toythereum

toythereum has the toy solidity compiler and evm.
It's not fully functioned like solc and geth, but it can compile solidity-like language and you can run the bytecode in the evm.

## how to use

1. write your solidity code, and put the file into contracts dir
   but it accept only uint256, bool, and string so far.
   Additionally, it doesn't have public, private, view, and etc..
   See example.sol.

2. compile your contract

```
$ sh cmd/compile_contract.sh your_code.sol
```

Then, a bytecode will be generated in the bytecode dir.

3. run evm

```
$ sh cmd/run_evm.sh
```

The process accepts your input. You can input your transaction input json. (See transactions dir)

4. deploy your contract
   At first, you have to create a json file for your deployment like,

```json
{
  "from": "0x1234567890abcdef1234567890abcdef12345678",
  "gas_limit": "21000",
  "gas_price": "10",
  "nonce": "0",
  "value": "10000000000",
  "data": "your byte code for the deployment here"
}
```

In data field, you have to input the bytecode you generated in section 2.

Then, input the file name into the running evm. You can see the contract address and the contract bytecode.

5. send transactions
   Like the previous deployment, set a json file and input the file name in the evm.
   But, you need to property and the first 4 bytes of the data field is function selector.

you can calculate function selector like below.

```
$ sh cmd/calc_selector.sh "compare(uint256)"

Function selector: "c4b8c257"
```

Then, constract input data. The first 4 bytes is the selector, and add following arguments.
For example, in the function `compare(u256)`, the selector is `c4b8c257` and the argument u256 is 5 here, which is `0x00000000000000000000000000000005`.
So the whole input data will be `0xc4b8c25700000000000000000000000000000005`

```json
{
  "from": "0x1234567890abcdef1234567890abcdef12345678",
  "to": "0x8770969cc5e7ffb9b76da997b97073cc83dbf142",
  "gas_limit": "21000",
  "gas_price": "10",
  "nonce": "0",
  "value": "10000000000",
  "data": "0xc4b8c25700000000000000000000000000000005"
}
```

# info from ethereum yellow paper

## 9.1 evm basics

- stack based architecture
- stack max size: 1024
- word size: 256bit
- memory model: word-addressed byte array
- storage model: word-addressed word array
- not following von Neumann architecture
- program code is stored separately in a virtual ROM

## 9.3 Execution Environment

necessary information to change the state

- current state: σ
- remaining gas for computation: g
- other information I
  - I_a: contract address
  - I_o: original sender address
  - I_p: gas price
  - I_d: input data byte array
  - I_s: sender address
  - I_v: value in wei
  - I_H: the block header of the parent block
  - I_e: the depth of the present message-call or contract creation
  - I_w: the permission to make modifications to the state
