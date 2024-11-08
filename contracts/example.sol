contract TestContract {
    uint256 a = 1 + 2 * 3 + 4;

    function test() returns (uint256) {
        uint256 b = 2 + a;
        return b;
    }

    function test2(uint256 c) returns (bool) {
        return a < c;
    }
}
