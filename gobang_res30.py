from tkinter import *
import math
import pickle
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_res_tensorflow import PolicyValueNet # Tensorflow
from human_play import Human

#定义棋盘类
class chessBoard() :
    def __init__(self, **kwargs) :
        self.width = int(kwargs.get('width', 9))
        self.height = int(kwargs.get('height', 9))
        # need how many pieces in a row to win
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.row = self.height - 1
        self.column = self.width - 1

        self.window = Tk()
        self.window.title("五子棋游戏")
        self.window.geometry("660x450")
        self.window.resizable(0,0)
        self.canvas=Canvas(self.window , bg="#EEE8AC" , width=self.column*50+50, height=self.row*50+50)
        self.paint_board()
        self.canvas.grid(row=0, column=0)

    def paint_board(self) :
        for row in range(0, self.height):
            if row == 0 or row == self.row:
                self.canvas.create_line(25, 25+row*50, 25+self.row*50, 25+row*50, width=2)
            else :
                self.canvas.create_line(25, 25+row*50, 25+self.row*50, 25+row*50, width=1)
        
        for column in range(0, self.width):
            if column == 0 or column == self.column:
                self.canvas.create_line(25+column*50, 25, 25+column*50, 25+self.column*50, width=2)
            else :
                self.canvas.create_line(25+column*50, 25, 25+column*50, 25+self.column*50, width=1)
        column = self.column // 4
        row = self.row // 4
        x = 25+column*50
        y = 25+row*50
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        x = 25+(self.column-column)*50
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        y = 25+(self.row-row)*50
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        x = 25+column*50
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
        x = 25+(self.column//2)*50
        y = 25+(self.row//2)*50
        self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")


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

        self.start_player = 0
        width, height, n_in_row = 9, 9, 5
        model_file = 'output/best_policy.model'
        board = Board(width=width, height=height, n_in_row=n_in_row)
        self.game = Game(board)
        self.game.board.init_board(self.start_player)
        self.best_policy = PolicyValueNet(width, height, model_file=model_file)
        self.mcts_player = MCTSPlayer(self.best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)  # set larger n_playout for better performance
        self.human_player = Human()
        self.human_player.set_player_ind(1)
        self.mcts_player.set_player_ind(2)
        self.players = {1:self.human_player, 2:self.mcts_player}

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
            if len(self.order) > 0:
                last_move = self.order[-1]
                last_y = last_move//9
                last_x = last_move%9
                self.change_color()
                self.board.canvas.delete(f"chessman{last_move}")
                self.board.canvas.create_oval(25+50*last_x-15 , 25+50*last_y-15 , 25+50*last_x+15 , 25+50*last_y+15 , fill = self.color,tags = f"chessman{last_move}")
                self.change_color()

            self.db[y][x] = self.color_count
            current_move = x+9*y
            self.order.append(current_move)
            self.board.canvas.create_oval(25+50*x-18 , 25+50*y-18 , 25+50*x+18 , 25+50*y+18 , fill = self.color,tags = f"chessman{current_move}")
            current_player = self.game.board.get_current_player()
            player_in_turn = self.players[current_player]
            print(self.color, player_in_turn.__class__.__name__, f"{x}, {y}")
            self.game.board.do_move(current_move)
            end, winner = self.game.board.game_end()
            if end:
                self.flag_win = 1
                self.flag_empty = 0
                print(self.color," win!!!")
                self.game_print.set(self.color+"获胜")
            else:
                self.change_color()
                self.game_print.set("请"+self.color+"落子")
                current_player = self.game.board.get_current_player()
                if current_player == self.human_player.player:
                    return
                self.board.window.update()
                player_in_turn = self.players[current_player]
                move = player_in_turn.get_action(self.game.board)
                x = move%9
                y = move//9
    

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
        x = z%9
        y = z//9
        self.db[y][x] = 2
        self.color_count = 1
        for i in self.order :
            ix = i%9
            iy = i//9
            self.change_color()
            self.board.canvas.create_oval(25+50*ix-15 , 25+50*iy-15 , 25+50*ix+15 , 25+50*iy+15 , fill = self.color,tags = "chessman")
        self.change_color()
        self.game_print.set("请"+self.color+"落子")
    

    #清空
    def empty_all(self) :
        self.board.canvas.delete("chessman")
        #还原初始化
        self.db = [([2] * 9) for i in range(9)]
        self.order = []
        self.color_count = 0 
        self.color = 'black'
        self.flag_win = 1
        self.flag_empty = 1
        self.game_print.set("")
        self.start_player = (self.start_player+1)%2
        self.game.board.init_board(self.start_player)


    #将self.flag_win置0才能在棋盘上落子
    def game_start(self) :
        #没有清空棋子不能置0开始
        if self.flag_empty == 0:
            return
        self.flag_win = 0
        self.game_print.set("请"+self.color+"落子")
        
        current_player = self.game.board.get_current_player()
        if current_player == self.human_player.player:
            return

        player_in_turn = self.players[current_player]
        move = player_in_turn.get_action(self.game.board)
        x = move%9
        y = move//9    
        self.db[y][x] = self.color_count
        self.order.append(move)
        self.board.canvas.create_oval(25+50*x-18 , 25+50*y-18 , 25+50*x+18 , 25+50*y+18 , fill = self.color,tags = f"chessman{move}")
        print(self.color, player_in_turn.__class__.__name__, f"{x}, {y}")
        self.game.board.do_move(move)
        self.change_color()
        self.game_print.set("请"+self.color+"落子")
        self.board.window.update()


    def options(self) :
        self.board.canvas.bind("<Button-1>",self.chess_moving)
        Label(self.board.window , textvariable = self.game_print , font = ("Arial", 20) ).place(relx = 0, rely = 0 ,x = 475 , y = 200)
        Button(self.board.window , text= "开始游戏" ,command = self.game_start,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=475, y=15)
        #Button(self.board.window , text= "我要悔棋" ,command = self.withdraw,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=475, y=60)
        Button(self.board.window , text= "清空棋局" ,command = self.empty_all,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=475, y=105)
        Button(self.board.window , text= "结束游戏" ,command = self.board.window.destroy,width = 13, font = ("Verdana", 12)).place(relx=0, rely=0, x=475, y=400)
        self.board.window.mainloop()

    
if __name__ == "__main__":
    game = Gobang()