# pytorch手写transformer

为了熟悉transformer架构，参照the annotated transformer自己写了一个transformer。

由于没有GPU，就简单的跑了个copy task，效果不是很好，和the annotated transformer的结果比较差，模型总是倾向于重复输出一样的token，可能是训练的逻辑有问题...

- linear.py: 简单学习了一下pytorch搭模型的流程，写了个简单的线性层模型
- attention.ipynb: 学习d2l写的attention
- transformer.ipynb: transformer的实现和简单训练
- torch_transformer.py: 直接使用pytorch自带的transformer做了个简单的训练
- the_annotated_transformer.py: the annotated transformer原始代码
