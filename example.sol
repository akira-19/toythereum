contract TestContract {
    uint256 a = 1;

    function test() returns (uint256) {
        uint256 b = 2 + a;
        return b;
    }
}
