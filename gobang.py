from tkinter import *
import math 

#定义棋盘类
class chessBoard() :
    def __init__(self) :
        self.window = Tk()
        self.window.title("五子棋游戏")
        self.window.geometry("660x500")
        self.window.resizable(0,0)
        self.canvas=Canvas(self.window , bg="#EEE8AC" , width=500, height=500)
        self.paint_board()
        self.canvas.grid(row=0, column=0)

    def paint_board(self) :
        for row in range(0, 10):
            if row == 0 or row == 9:
                self.canvas.create_line(25, 25+row*50, 25+9*50, 25+row*50, width=2)
            else :
                self.canvas.create_line(25, 25+row*50, 25+9*50, 25+row*50, width=1)
        
        for column in range(0, 10):
            if column == 0 or column == 9:
                self.canvas.create_line(25+column*50, 25, 25+column*50, 25+9*50, width=2)
            else :
                self.canvas.create_line(25+column*50, 25, 25+column*50, 25+9*50, width=1)
            
        self.canvas.create_oval(122, 122, 128, 128, fill="black")
        self.canvas.create_oval(372, 122, 378, 128, fill="black")
        self.canvas.create_oval(122, 372, 128, 378, fill="black")
        self.canvas.create_oval(372, 372, 378, 378, fill="black")


#定义五子棋游戏类
#0为黑子 ， 1为白子 ， 2为空位
class Gobang() :
    #初始化
    def __init__(self) :
        self.board = chessBoard()
        self.game_print = StringVar()
        self.game_print.set("")
        #16*16的二维列表，保证不会out of index
        self.db = [([2] * 9) for i in range(9)]
        #悔棋用的顺序列表
        self.order = []
        #棋子颜色
        self.color_count = 0 
        self.color = 'black'
        #清空与赢的初始化，已赢为1，已清空为1
        self.flag_win = 1
        self.flag_empty = 1
        self.options()
        
     
    #黑白互换
    def change_color(self) :
        self.color_count = (self.color_count + 1 ) % 2
        if self.color_count == 0 :
            self.color = "black"
        elif self.color_count ==1 :
            self.color = "white"
    
    
    #落子
    def chess_moving(self ,event) :
        #不点击“开始”与“清空”无法再次开始落子
        if self.flag_win ==1 or self.flag_empty ==0:
            return
        #坐标转化为下标
        x,y = event.x-25 , event.y-25
        x = round(x/50)
        y = round(y/50)
        #点击位置没用落子，且没有在棋盘线外，可以落子
        while self.db[y][x] == 2 and self.limit_boarder(y,x):
            self.db[y][x] = self.color_count
            self.order.append(x+15*y)   
            self.board.canvas.create_oval(25+50*x-15 , 25+50*y-15 , 25+50*x+15 , 25+50*y+15 , fill = self.color,tags = "chessman")
            if self.game_win(y,x,self.color_count) :
                print(self.color,"获胜")
                self.game_print.set(self.color+"获胜")
            else :
                self.change_color()
                self.game_print.set("请"+self.color+"落子")
    

    #保证棋子落在棋盘上
    def limit_boarder(self , y , x) :
        if x<0 or x>8 or y<0 or y>8 :
            return False
        else :
            return True


    #计算连子的数目,并返回最大连子数目
    def chessman_count(self , y , x , color_count ) :
        count1,count2,count3,count4 = 1,1,1,1
        #横计算
        for i in range(-1 , -5 , -1) :
            if self.db[y][x+i] == color_count  :
                count1 += 1
            else:
                break 
        for i in  range(1 , 5 ,1 ) :
            if self.db[y][x+i] == color_count  :
                count1 += 1
            else:
                break 
        #竖计算
        for i in range(-1 , -5 , -1) :
            if self.db[y+i][x] == color_count  :
                count2 += 1
            else:
                break 
        for i in  range(1 , 5 ,1 ) :
            if self.db[y+i][x] == color_count  :
                count2 += 1
            else:
                break 
        #/计算
        for i in range(-1 , -5 , -1) :
            if self.db[y+i][x+i] == color_count  :
                count3 += 1
            else:
                break 
        for i in  range(1 , 5 ,1 ) :
            if self.db[y+i][x+i] == color_count  :
                count3 += 1
            else:
                break 
        #\计算
        for i in range(-1 , -5 , -1) :
            if self.db[y+i][x-i] == color_count :
                count4 += 1
            else:
                break 
        for i in  range(1 , 5 ,1 ) :
            if self.db[y+i][x-i] == color_count :
                count4 += 1
            else:
                break 
            
        return max(count1 , count2 , count3 , count4)


    #判断输赢
    def game_win(self , y , x , color_count ) :
        if self.chessman_count(y,x,color_count) >= 5 :
            self.flag_win = 1
            self.flag_empty = 0
            return True
        else :
            return False
        

    #悔棋,清空棋盘，再画剩下的n-1个棋子
    def withdraw(self ) :
        if len(self.order)==0 or self.flag_win == 1:
            return
        self.board.canvas.delete("chessman")
        z = self.order.pop()
        x = z%15
        y = z//15
        self.db[y][x] = 2
        self.color_count = 1
        for i in self.order :
            ix = i%15
            iy = i//15
            self.change_color()
            self.board.canvas.create_oval(25+50*ix-15 , 25+50*iy-15 , 25+50*ix+15 , 25+50*iy+15 , fill = self.color,tags = "chessman")
        self.change_color()
        self.game_print.set("请"+self.color+"落子")
    

    #清空
    def empty_all(self) :
        self.board.canvas.delete("chessman")
        #还原初始化
        self.db = [([2] * 16) for i in range(16)]
        self.order = []
        self.color_count = 0 
        self.color = 'black'
        self.flag_win = 1
        self.flag_empty = 1
        self.game_print.set("")



    #将self.flag_win置0才能在棋盘上落子
    def game_start(self) :
        #没有清空棋子不能置0开始
        if self.flag_empty == 0:
            return
        self.flag_win = 0
        self.game_print.set("请"+self.color+"落子")


    def options(self) :
        self.board.canvas.bind("<Button-1>",self.chess_moving)
        Label(self.board.window , textvariable = self.game_print , font = ("Arial", 20) ).place(relx = 0, rely = 0 ,x = 505 , y = 200)
        Button(self.board.window , text= "开始游戏" ,command = self.game_start,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=505, y=15)
        Button(self.board.window , text= "我要悔棋" ,command = self.withdraw,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=505, y=60)
        Button(self.board.window , text= "清空棋局" ,command = self.empty_all,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=505, y=105)
        Button(self.board.window , text= "结束游戏" ,command = self.board.window.destroy,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=505, y=420)
        self.board.window.mainloop()

    
if __name__ == "__main__":
    game = Gobang()