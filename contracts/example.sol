contract TestContract {
    uint256 a = 1 + 2 * 3 + 4;

    function increment() returns (uint256) {
        a = a + 1;
        return a;
    }

    function compare(uint256 b) returns (bool) {
        return a < b;
    }
}
