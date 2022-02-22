

class wallet:
    def __int__(self):
        self.cash = 1000
        self.bit = 0
        self.gold = 0
        self.alphaGold = 0.01
        self.alphaBit = 0.02

    def buy_gold(self,num_gold, gold_price):
        if self.cash >= num_gold * gold_price + num_gold * gold_price * self.alphaGold:
            self.gold += num_gold
            self.cash -= num_gold * gold_price + num_gold * gold_price * self.alphaGold
        else:
            raise ValueError("cash not enough while buying gold")

    def buy_bit(self,num_bit,bit_price):
        if self.cash >= num_bit * bit_price + num_bit * bit_price * self.alphaBit:
            self.gold += num_bit
            self.cash -= num_bit * bit_price + num_bit * bit_price * self.alphaBit
        else:
            raise ValueError("cash not enough while buying bit coin")

    def sell_gold(self,num_gold, gold_price):
        if self.gold * gold_price >= num_gold * gold_price + num_gold * gold_price * self.alphaGold:
            self.gold -= num_gold
            self.cash += num_gold * gold_price - num_gold * gold_price * self.alphaGold
        else:
            raise ValueError("gold not enough while selling gold")

    def sell_bit(self,num_bit, bit_price):
        if self.gold * bit_price >= num_bit * bit_price + num_bit * bit_price * self.alphaBit:
            self.gold -= num_bit
            self.cash += num_bit * bit_price - num_bit * bit_price * self.alphaBit
        else:
            raise ValueError("bit coin not enough while selling bit coin")

    def cal_bit_fee(self,num_bit, bit_price):
        return num_bit * bit_price * self.alphaBit

    def cal_gold_fee(self,num_gold, gold_price):
        return num_gold * gold_price * self.alphaGold