module Semiring where

class Semiring x where
    zero :: x
    one :: x
    plus :: x-> x -> x
    prod :: x-> x -> x

instance Semiring Bool where  
  zero = False
  one = True
  plus = (||) 
  prod = (&&)

instance Semiring Int where  
  zero = 0
  one = 1
  plus = (+)
  prod = (*)

instance Semiring Integer where  
  zero = 0
  one = 1
  plus = (+)
  prod = (*)

instance Semiring Double where  
  zero = 0
  one = 1
  plus = (+)
  prod = (*)

instance Semiring Float where  
  zero = 0
  one = 1
  plus = (+)
  prod = (*)