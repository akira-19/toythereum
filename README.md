# toythereum

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

## comments
