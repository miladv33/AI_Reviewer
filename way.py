import random
from datetime    import datetime
class a:
    def __init__(self,n,a):
        self.name=n
        self.age=a
        self.grades=[]
    def addgrade(self,g):
                self.grades.append(g)
class management():
    def __init__(self):
        self.students=[]
        self.MAXSTUDENTS=50
    def add(self,name,age):
        if len(self.students)<self.MAXSTUDENTS:
         x=a(name,age)
         self.students.append(x)
         return True
        else:
            return False
    def calculate_average(self):
        total=0
        count=0
        for i in self.students:
            for j in i.grades:
                total=total+j
                count=count+1
        if count>0:
            return total/count
        else:
            return 0
    def find(self,name):
        for x in range(0,len(self.students)):
            if self.students[x].name==name:
                return x
        return -1
    def Remove(self,name):
        i = self.find(name)
        if i >= 0:
            self.students.pop(i)
            return True
        return False
    def ADD_GRADE(self,name,grade):
        index=self.find(name)
        if index>=0:
            self.students[index].addgrade(grade)
            return True
        return False
    def get_student_info(self,name):
        i=self.find(name)
        if i>=0:
            return {"name":self.students[i].name,"age":self.students[i].age,"grades":self.students[i].grades}
        return None
    def generate_report(self):
        report = ""
        for s in self.students:
            report += f"Student: {s.name}\n"
            report += f"Age: {s.age}\n"
            if len(s.grades) > 0:
                avg = sum(s.grades)/len(s.grades)
            else:
                avg = 0
            report += f"Average Grade: {avg}\n"
            report += "-" * 20 + "\n"
        return report

def main():
    mgmt=management()
    mgmt.add("john doe",20)
    mgmt.ADD_GRADE("john doe",85)
    mgmt.ADD_GRADE("john doe",90)
    mgmt.ADD_GRADE("john doe",78)
    mgmt.add("jane smith",19)
    mgmt.ADD_GRADE("jane smith",92)
    mgmt.ADD_GRADE("jane smith",88)
    print(mgmt.calculate_average())
    print(mgmt.generate_report())
    mgmt.Remove("john doe")
    print(mgmt.get_student_info("jane smith"))

if __name__=="__main__":
    main()