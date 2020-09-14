from datetime import datetime
from django.shortcuts import render, redirect
# Create your views here.
from .models import Board, Answers, Admin
from datetime import datetime, timedelta, date
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.contrib.auth import login


import post_detec_bot
import ReinforcementLearning.reinforce
import time

post_detec_bot.detec_bot()
def index(request):
    if request.session['user'] == 'admin':
        return redirect('/admin_main')
    else:
        print(request.session['user'])
        return redirect('/')

def main(request):
    request.session['user'] = 'user'
    rs_board = Board.objects.all().order_by('-no')
    paginator = Paginator(rs_board, 10)
    page=request.GET.get('page')
    # ?page=1
    rs_board = paginator.get_page(page)
    return render(request, "main.html", {"rs_board": rs_board})

def admin_main(request):
    user_info = request.session['user']
    print(user_info)
    rs_board = Board.objects.all().order_by('-no')
    paginator = Paginator(rs_board, 10)
    page = request.GET.get('page')
    # ?page=1
    rs_board = paginator.get_page(page)
    return render(request, "admin_main.html", {"rs_board": rs_board})

def log_in(request):
    if request.method=="GET":
        return render(request,'/')
    if request.method=="POST":
        admin_id = request.POST["user_id"]
        admin_pw = request.POST["user_pw"]
        admin = Admin.objects.filter(admin_id=admin_id, admin_pw=admin_pw)
        if admin is not None:
            request.session['user'] = admin_id
            return redirect('/index')
        else:

            return redirect('/index')

def log_out(request):
    request.session['user'] = "user"
    return redirect('/index')

def board_write(request):
    return render(request, "board_write.html", )

def board_insert(request):
    user_info = request.session['user']
    print(user_info)
    writer = request.GET['writer']
    question = request.GET['question']
    pw = request.GET['password']
    print(pw)
    question = question.upper()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if question != "" and writer != "" and pw != "":
        row = Board.objects.create(writer=writer, question=question, date=date, pw=pw)
        return redirect('/index')
    else:
        return redirect('/board_write')

def delete(request):
    no = request.GET['no']
    row=Board.objects.filter(no=no)
    row.delete()
    rows=Answers.objects.all().filter(board_no=no)
    for row in rows:
        row.delete()
    return redirect('/')

def board_detail(request, no):
    # no=request.GET['no']
    rsDetail = Board.objects.filter(no=no)
    rsAnswers = Answers.objects.filter(board_no=no)
    return render(request, "detail.html", {'rsDetail': rsDetail, 'rsAnswers': rsAnswers})


def rating_insert(request):
    board_no = request.GET['board_no']
    answer_no = request.GET['answer_no']

    answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)
    print(answer_row.good)

    if request.GET['rating'] == 'good':

        if answer_row.good is None:
            answer_row.good = 0

        answer_row.good = answer_row.good + 1
        answer_row.save()
        return redirect('board_detail', board_no)

    elif request.GET['rating'] == 'bad':

        if answer_row.bad is None:
            answer_row.bad = 0

        answer_row.bad = answer_row.bad + 1
        answer_row.save()
        return redirect('board_detail', board_no)

#----------------------관리자 기능--------------------

def reinforce_view(request):
    answers = Answers.objects.all().order_by('-good','-board_no')
    boards = Board.objects.all().order_by('-no')
    paginator = Paginator(answers, 20)
    page = request.GET.get('page')
    # ?page=1
    answers = paginator.get_page(page)
    return render(request, "reinforce.html",{"answers": answers , "boards" : boards})

def delete_admin(request):
    board_no_arr = request.GET.getlist('cb')
    print(board_no_arr)

    for i in board_no_arr:
        board = Board.objects.get(no=i)
        board.delete()
        rows = Answers.objects.all().filter(board_no=i)
        for row in rows:
            row.delete()
    #rs_board = Board.objects.all().order_by('-no')
    return redirect('/admin_main')

def function_answer(request):
    board_no_arr = request.GET.getlist('cb')
    print(board_no_arr)

    if request.GET['function'] == 'delete':
        for i in board_no_arr:
            board_no = i.split('+')[0]  # board_no
            answer_no = i.split("+")[1]  #
            answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)
            answer_row.delete()

    elif request.GET['function'] == 'reinforce':
        # 파일 저장 경로와 이름, 나중에 경로 수정 필요
        # timestr = time.strftime("%Y%m%d-%H%M%S")
        f = open("C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/SelectedAnswerSet.txt", 'w+', -1, encoding='UTF-8')

        for i in board_no_arr:
            board_no = i.split('+')[0]  # board_no
            answer_no = i.split("+")[1]  #
            answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)
            question = Board.objects.get(no=board_no).question
            answer = answer_row.answer  # 답
            article_url = answer_row.article_url  # url
            article_title = answer_row.article_title  # title
            print(question)
            print(answer)
            print(article_url)
            print(article_title)

            # URL 이용해서 엘라스틱서치로 본문 가져오기
            context = ReinforcementLearning.reinforce.get_content_by_URL(article_url)

            line = article_title + '§' + context + '§' + question + '§' + answer + '§' + str(
                context.find(answer)) + '\n'
            f.write(line)

        f.close()

        # txt -> json / 파일 경로 수정 필요
        ReinforcementLearning.reinforce.create_json("C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/SelectedAnswerSet.txt")
    return redirect('reinforce_view')



