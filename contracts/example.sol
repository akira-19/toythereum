contract TestContract {
    uint256 a = 1 + 2 * 3 + 4;

    function increment() returns (uint256) {
        uint256 b = a + 1;
        return b;
    }

    function compare(uint256 c) returns (bool) {
        return a < c;
    }
}
