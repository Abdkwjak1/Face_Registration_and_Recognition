import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont
import Fc_Rec
from csv import DictWriter
import os
import csv

class App():
    def __init__(self, root):
        self.root=root
        self.FR=Fc_Rec.Fc_Rec()

        #setting title
        root.title("Face ID")
        #setting window size
        width=307
        height=292
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        self.name_var=tk.StringVar()
        self.phone_num_var=tk.StringVar()
        self.res_day_var=tk.StringVar()
        self.date_res_var=tk.StringVar()
        self.room_num_var=tk.StringVar()
        self.floor_num_var=tk.StringVar()
        self.room_cap_var=tk.StringVar()
        self.ID=0
        

        GLabel_648=tk.Label(root)
        ft = tkFont.Font(family='Times',size=14)
        GLabel_648["font"] = ft
        GLabel_648["fg"] = "#333333"
        GLabel_648["justify"] = "center"
        GLabel_648["text"] = "Select one:"
        GLabel_648.place(x=0,y=0,width=87,height=36)

        GButton_380=tk.Button(root)
        GButton_380["bg"] = "#000000"
        ft = tkFont.Font(family='Times',size=11)
        GButton_380["font"] = ft
        GButton_380["fg"] = "#ffffff"
        GButton_380["justify"] = "center"
        GButton_380["text"] = "Test The Camera"
        GButton_380.place(x=10,y=250,width=127,height=31)
        GButton_380["command"] = self.GButton_380_command

        GButton_579=tk.Button(root)
        GButton_579["anchor"] = "center"
        GButton_579["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_579["font"] = ft
        GButton_579["fg"] = "#ffffff"
        GButton_579["justify"] = "center"
        GButton_579["text"] = "Enter New Guest"
        GButton_579["relief"] = "flat"
        GButton_579.place(x=180,y=30,width=114,height=30)
        GButton_579["command"] = self.GButton_579_command

        GButton_748=tk.Button(root)
        GButton_748["anchor"] = "center"
        GButton_748["bg"] = "#999999"
        ft = tkFont.Font(family='Times',size=10)
        GButton_748["font"] = ft
        GButton_748["fg"] = "#ffffff"
        GButton_748["justify"] = "center"
        GButton_748["text"] = "Check Geust"
        GButton_748["relief"] = "flat"
        GButton_748.place(x=180,y=80,width=114,height=30)
        GButton_748["command"] = self.GButton_748_command


    def GButton_380_command(self):
        self.FR.Test_Camera()


    def GButton_579_command(self):
        self.App1(self.root)  


    def GButton_748_command(self):
        id=self.FR.detect()

        with open('Data.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            #header = next(reader)
            #if header != None:
            for row in reader:
                if int(row['ID']) == id:
                    guest=row
                    break

        self.App2(self.root,guest)



    def App2(self,root1,guest):
        root=tk.Toplevel(root1)
        #setting title
        root.title("Guest Info")
        #setting window size
        width=408
        height=373
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)


        name_label = tk.Label(root, text = 'Full Name:', font=('calibre',10, 'bold'))
        name_entry = tk.Label(root, text = guest['Name'], font=('calibre',10,'normal'))

        phone_num_label = tk.Label(root, text = 'Phone Number:', font=('calibre',10, 'bold'))
        phone_num_entry = tk.Label(root, text = guest['Phone'], font=('calibre',10,'normal'))

        res_day_label = tk.Label(root, text = 'Reservation Days:', font=('calibre',10, 'bold'))
        res_day_entry = tk.Label(root,text = guest['Res_D'], font=('calibre',10,'normal'))

        date_res_label = tk.Label(root, text = 'Date of Reservation:', font=('calibre',10, 'bold'))
        date_res_entry = tk.Label(root,text = guest['Date_R'], font=('calibre',10,'normal'))

        floor_num_label = tk.Label(root, text = 'Floor Number:', font=('calibre',10, 'bold'))
        floor_num_entry = tk.Label(root,text = guest['Floor_N'], font=('calibre',10,'normal'))

        room_num_label = tk.Label(root, text = 'Room Number:', font=('calibre',10, 'bold'))
        room_num_entry = tk.Label(root,text = guest['Room_N'], font=('calibre',10,'normal'))

        room_cap_label = tk.Label(root, text = 'Room Capacity:', font=('calibre',10, 'bold'))
        room_cap_entry = tk.Label(root,text = guest['Room_C'], font=('calibre',10,'normal'))

        name_label.place(x=10,y=20,width=160,height=30)
        name_entry.place(x=170,y=20,width=160,height=30)

        phone_num_label.place(x=10,y=60,width=160,height=30)
        phone_num_entry.place(x=170,y=60,width=160,height=30)

        res_day_label.place(x=10,y=100,width=160,height=30)
        res_day_entry.place(x=170,y=100,width=160,height=30)

        date_res_label.place(x=10,y=140,width=160,height=30)
        date_res_entry.place(x=170,y=140,width=160,height=30)

        floor_num_label.place(x=10,y=180,width=160,height=30)
        floor_num_entry.place(x=170,y=180,width=160,height=30)

        room_num_label.place(x=10,y=220,width=160,height=30)
        room_num_entry.place(x=170,y=220,width=160,height=30)

        room_cap_label.place(x=10,y=260,width=160,height=30)
        room_cap_entry.place(x=170,y=260,width=160,height=30)

       



    def App1(self,root1):
        root=tk.Toplevel(root1)
        #setting title
        root.title("Guest Info")
        #setting window size
        width=408
        height=373
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)


        name_label = tk.Label(root, text = 'Full Name', font=('calibre',10, 'bold'))
        name_entry = tk.Entry(root,textvariable = self.name_var, font=('calibre',10,'normal'))

        phone_num_label = tk.Label(root, text = 'Phone Number', font=('calibre',10, 'bold'))
        phone_num_entry = tk.Entry(root,textvariable = self.phone_num_var, font=('calibre',10,'normal'))

        res_day_label = tk.Label(root, text = 'Reservation Days', font=('calibre',10, 'bold'))
        res_day_entry = tk.Entry(root,textvariable = self.res_day_var, font=('calibre',10,'normal'))

        date_res_label = tk.Label(root, text = 'Date of Reservation', font=('calibre',10, 'bold'))
        date_res_entry = tk.Entry(root,textvariable = self.date_res_var, font=('calibre',10,'normal'))

        floor_num_label = tk.Label(root, text = 'Floor Number', font=('calibre',10, 'bold'))
        floor_num_entry = tk.Entry(root,textvariable = self.floor_num_var, font=('calibre',10,'normal'))

        room_num_label = tk.Label(root, text = 'Room Number', font=('calibre',10, 'bold'))
        room_num_entry = tk.Entry(root,textvariable = self.room_num_var, font=('calibre',10,'normal'))

        room_cap_label = tk.Label(root, text = 'Room Capacity', font=('calibre',10, 'bold'))
        room_cap_entry = tk.Entry(root,textvariable = self.room_cap_var, font=('calibre',10,'normal'))

        name_label.place(x=10,y=20,width=160,height=30)
        name_entry.place(x=170,y=20,width=160,height=30)

        phone_num_label.place(x=10,y=60,width=160,height=30)
        phone_num_entry.place(x=170,y=60,width=160,height=30)

        res_day_label.place(x=10,y=100,width=160,height=30)
        res_day_entry.place(x=170,y=100,width=160,height=30)

        date_res_label.place(x=10,y=140,width=160,height=30)
        date_res_entry.place(x=170,y=140,width=160,height=30)

        floor_num_label.place(x=10,y=180,width=160,height=30)
        floor_num_entry.place(x=170,y=180,width=160,height=30)

        room_num_label.place(x=10,y=220,width=160,height=30)
        room_num_entry.place(x=170,y=220,width=160,height=30)

        room_cap_label.place(x=10,y=260,width=160,height=30)
        room_cap_entry.place(x=170,y=260,width=160,height=30)

        GButton_434=tk.Button(root)
        GButton_434["bg"] = "#f71515"
        ft = tkFont.Font(family='Times',size=10)
        GButton_434["font"] = ft
        GButton_434["fg"] = "#f9f8f8"
        GButton_434["justify"] = "center"
        GButton_434["text"] = "Submit "
        GButton_434.place(x=10,y=320,width=106,height=30)
        GButton_434["command"] = self.GButton_434_command

        GButton_356=tk.Button(root)
        GButton_356["bg"] = "#f71515"
        ft = tkFont.Font(family='Times',size=10)
        GButton_356["font"] = ft
        GButton_356["fg"] = "#f9f8f8"
        GButton_356["justify"] = "center"
        GButton_356["text"] = "Capture Face "
        GButton_356.place(x=120,y=320,width=106,height=30)
        GButton_356["command"] = self.GButton_356_command


        GButton_350=tk.Button(root)
        GButton_350["bg"] = "#000000"
        ft = tkFont.Font(family='Times',size=10)
        GButton_350["font"] = ft
        GButton_350["fg"] = "#ffffff"
        GButton_350["justify"] = "center"
        GButton_350["text"] = "Click before Exit "
        GButton_350.place(x=260,y=320,width=124,height=30)
        GButton_350["command"] = self.GButton_350_command



    def GButton_434_command(self):
        name=self.name_var.get()
        phone_num=self.phone_num_var.get()
        res_day=self.res_day_var.get()
        date_res=self.date_res_var.get()
        room_num=self.room_num_var.get()
        floor_num=self.floor_num_var.get()
        room_cap=self.room_cap_var.get()


        if os.path.isfile("Data.csv"):
            with open('Data.csv', 'r') as f:
                reader = csv.reader(f)
                self.ID=len(list(reader))

        #write to csv file code here
        with open('Data.csv', 'a+', newline = '') as f:
            
            dict_writer = DictWriter(f, fieldnames=['Name', 'Phone', 'Res_D','Date_R', 'Floor_N', 'Room_N', 'Room_C','ID'])
            if os.stat('Data.csv').st_size == 0:        #if file is not empty than header write else not
                dict_writer.writeheader()
                self.ID=1
       
            dict_writer.writerow({
                'Name' : name,
                'Phone' : str(phone_num),
                'Res_D' : res_day,
                'Date_R' : date_res,
                'Floor_N' : floor_num,
                'Room_N' : room_num,
                'Room_C' : room_cap,
                'ID': str(self.ID)
            })



        self.name_var.set("")
        self.phone_num_var.set("")
        self.res_day_var.set("")
        self.date_res_var.set("")
        self.room_num_var.set("")
        self.floor_num_var.set("")
        self.room_cap_var.set("")

    def GButton_356_command(self):
        self.FR.register(self.ID)

    def GButton_350_command(self):
        self.FR.training()

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
