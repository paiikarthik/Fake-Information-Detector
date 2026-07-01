class Solution(object):
    def minimumTotal(self, triangle):
        new=[]
        for num in triangle:
            for row in num:
                new.append(row)
            return min(new)
        
        

triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
c=Solution()
print(c.minimumTotal(triangle))
