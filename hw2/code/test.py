import numpy as np
import matplotli b . p y pl o t a s p l t
np . random . s e e d ( 5 4 6 )
def cal lambdaMax ( x , y ) :
mean y = np . mean ( y )
l ambd a a r r ay = np . z e r o s ( d )
for k in range ( d ) :
x k = x [ : , k ]
lambda k = 2 ∗ np . abs ( np .sum( x k ∗ ( y − mean y ) , 0 ) )
l ambd a a r r ay [ k ] = lambda k
return max( l ambd a a r r ay )
def c o o rd ( tuning , d el t a , w input , x , y ) :
w new = np . copy ( w inpu t )
w prev = np . copy ( w inpu t )
no = 1
c h e c k d el t a = True
pr int ( tunin g )
while c h e c k d el t a :
w prev = np . copy ( w new )
no += 1
b = np . mean ( y − np . dot ( x , w prev ) )
for j in range ( d ) :
a = 2 ∗ np .sum( x [ : , j ] ∗ ∗ 2 )
wj = np . d e l e t e ( w new , j )
x i j = np . d e l e t e ( x , j , a x i s =1)
w j x i j = np . matmul ( xi j , wj )
ck = 2 ∗ np . dot ( x [ : , j ] , y − ( b + w j x i j ) )
i f ck < −tunin g :
w new [ j ] = ( ck + tunin g ) / a
e l i f ck > tunin g :
w new [ j ] = ( ck − tunin g ) / a
e l s e :
w new [ j ] = 0
c h e c k d el t a = any( np . abs ( w new − w prev ) > d e l t a )
i f no > 3 0:
c h e c k d el t a = F al s e
pr int ( no , end=” , ” ) # c o u n t i n g i t e r a t i o n s , f o r r e f e r e n c e
return w new
## Genera te Data
n = 500
d = 1000
k = 100
sigma = 1
r a t e = 1. 5
d e l t a = 0. 0 0 1
x = np . random . randn ( n , d )
n oi s e = np . random . randn ( n ) ∗ ( sigma ∗ ∗2 )
w = np . z e r o s ( ( d ) )
for j in range ( k ) :
