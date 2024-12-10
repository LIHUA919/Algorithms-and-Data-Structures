import turtle

# 创建屏幕
screen = turtle.Screen()
screen.bgcolor("black")  # 背景色

# 创建蛇的头部
snake = turtle.Turtle()
snake.shape("square")  # 蛇的形状
snake.color("green")    # 蛇的颜色
snake.speed(10)         # 速度

# 画蛇
for _ in range(6):  # 画6个方块
    snake.stamp()   # 留下印记
    snake.forward(20)  # 向前移动20个单位
    snake.right(60)    # 向右转60度

# 结束
turtle.done()
