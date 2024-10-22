

class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int minprice = int(1e9);
        int maxprofit = 0;
        for (auto price : prices){
            maxprofit = max(maxprofit, price - minprice);
            minprice = min(minprice, price);
        }
        return maxprofit;
    }
};