## python面试

##### 1. 赋值 copy deepcopy区别

v = [1,[2,3],4]



##### 2.*args  **kwargs 使用



##### 3.生成器 

将列表生成式中[]改成() 之后数据结构是否改变？



##### 4. lambda filter map 

a = filter(lambda x:x>3, [1,2,3,4,5])

b = map(lambda x:x*x, [1,2,3])

​	type(a)?  

​	type(b)? 



##### 5.静态方法(staticmethod),类方法(classmethod)和实例方法

```python
def foo(x):
    print "executing foo(%s)"%(x)

class A(object):
    def foo(self,x):
        print("executing foo(%s,%s)"%(self,x))

    @classmethod
    def class_foo(cls,x):
        print("executing class_foo(%s,%s)"%(cls,x))

    @staticmethod
    def static_foo(x):
        print("executing static_foo(%s)"%x)

a=A()
```



##### 6.Python中单下划线和双下划线

`_foo`

`__foo`

`__foo__`



##### 7.算法

Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.

**Examples:**

```python
s = "leetcode"
return 0.

s = "loveleetcode",
return 2.
```





