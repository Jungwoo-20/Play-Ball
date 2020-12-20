from django.shortcuts import render, redirect
# Create your views here.
from .models import Board, Answers, Admin
from datetime import datetime
from django.http import HttpResponse
from django.core.paginator import Paginator
from django.db.models import Q
import post_detec_bot
import ReinforcementLearning.reinforce
import simplejson as json

post_detec_bot.detec_bot()


def index(request):

    if request.session['user'] == 'admin':
        return redirect('/admin_main')
    else:
        return redirect('/')


def main(request):
    request.session['user'] = 'user'
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('keyword', '')  # 검색어
    # 조회
    search_list = Board.objects.all().order_by('-no')
    if kw:
        search_list = search_list.filter(
            Q(question__icontains=kw) |
            Q(writer__icontains=kw)
        ).distinct()
    # 페이징처리
    paginator = Paginator(search_list, 10)
    page_obj = paginator.get_page(page)
    # ?page=1
    context = {"rs_board": page_obj, "page": page, "keyword": kw}
    return render(request, "main.html", context)


def admin_main(request):
    user_info = request.session['user']
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('keyword', '')  # 검색어
    # 조회
    search_list = Board.objects.all().order_by('-no')
    if kw:
        search_list = search_list.filter(
            Q(question__icontains=kw) |
            Q(writer__icontains=kw)
        ).distinct()
    paginator = Paginator(search_list, 10)
    page_obj = paginator.get_page(page)
    # ?page=1
    context = {"rs_board": page_obj, "page": page, "keyword": kw}
    return render(request, "admin_main.html", context)


def updateData(request):

    update_list = Board.objects.all().order_by('-no')[:10]
    user_info = request.session['user']

    if request.method == "GET":

        checkList = request.GET.getlist('cb')
        checkList = list(map(int, checkList))
        context = {"rs_board": update_list, "user_info": user_info, "checkList": checkList}
        return render(request, "updateData.html", context)
    else:
        context = {"rs_board": update_list, "user_info": user_info}
        return render(request, "updateData.html", context)


def log_in(request):
    if request.method == "GET":
        return render(request, '/')
    if request.method == "POST":
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
    writer = request.GET['writer']
    question = request.GET['question']
    pw = request.GET['password']
    question = question.upper()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if question != "" and writer != "" and pw != "":
        row = Board.objects.create(writer=writer, question=question, date=date, pw=pw)
        no = row.no
        param = str(no) + '!' + str(question)
        post_detec_bot.question_list.append(param)
        return redirect('/index')
    else:
        return redirect('/board_write')


def delete(request):
    no = request.GET['no']
    row = Board.objects.filter(no=no)
    row.delete()
    rows = Answers.objects.all().filter(board_no=no)
    for row in rows:
        row.delete()
    return redirect('/')


def board_detail(request, no):
    rsDetail = Board.objects.filter(no=no)
    rsAnswers = Answers.objects.filter(board_no=no)
    return render(request, "detail.html", {'rsDetail': rsDetail, 'rsAnswers': rsAnswers})


def rating_insert(request):
    board_no = request.GET['board_no']
    answer_no = request.GET['answer_no']

    answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)

    if request.GET['rating'] == 'good':

        if answer_row.good is None:
            answer_row.good = 0

        answer_row.good = answer_row.good + 1
        answer_row.save()

    elif request.GET['rating'] == 'bad':

        if answer_row.bad is None:
            answer_row.bad = 0

        answer_row.bad = answer_row.bad + 1
        answer_row.save()

    count = answer_row.good + answer_row.bad
    answer_row.ratio = (answer_row.good / count * 100)
    answer_row.save()
    return redirect('board_detail', board_no)


# ----------------------관리자 기능--------------------

def reinforce_view(request):
    page = request.GET.get('page', '1')  # 페이지
    kw = request.GET.get('keyword', '')  # 검색어

    # 조회
    answers_list = Answers.objects.all().order_by('-ratio', '-good', '-board_no')
    boards_list = Board.objects.all().order_by('-no')

    if kw:
        boards_list = boards_list.filter(question__icontains=kw)

    paginator = Paginator(answers_list, 20)
    page_obj = paginator.get_page(page)
    # ?page=1
    context = {"answers": page_obj, "boards": boards_list, "page": page, "keyword": kw}
    return render(request, "reinforce.html", context)

def Lookup(request):
    ratio = request.GET['goodratio']  # 비율
    acount = request.GET['Acount']  # 만족도 개수
    qcount = request.GET['Qcount']  # 선택할 답변

    # 조회
    answers_list = Answers.objects.all().exclude(reinforce=1)
    answers_list = answers_list.order_by('-ratio', '-good', '-board_no')
    boards_list = Board.objects.all().order_by('-no')

    answers_list = answers_list.filter(ratio__gte=int(ratio))

    answers = []
    for i in answers_list:
        count = i.good + i.bad

        if count >= int(acount):
            answers.append(i)

    if len(answers) > int(qcount):
        answers = answers[:int(qcount)]

    context = {"answers": answers, "boards": boards_list}

    return render(request, 'test.html', context)


def delete_admin(request):
    board_no_arr = request.GET.getlist('cb')

    for i in board_no_arr:
        board = Board.objects.get(no=i)
        board.delete()
        rows = Answers.objects.all().filter(board_no=i)
        for row in rows:
            row.delete()
    return redirect('/admin_main')


def function_answer(request):
    board_no_arr = request.GET.getlist('cb')

    if request.GET['function'] == 'delete':
        for i in board_no_arr:
            board_no = i.split('+')[0]  # board_no
            answer_no = i.split("+")[1]  #
            answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)
            answer_row.delete()
        return redirect('reinforce_view')

    elif request.GET['function'] == 'reinforce':
        answers_no_list = []
        f = open(
            "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/SelectedAnswerSet.txt",
            'w+', -1, encoding='UTF-8')

        for i in board_no_arr:
            board_no = i.split('+')[0]  # board_no
            answer_no = i.split("+")[1]  #
            answers_no_list.append(answer_no) #답변no 리스트
            answer_row = Answers.objects.get(board_no=board_no, answer_no=answer_no)
            question = Board.objects.get(no=board_no).question
            answer = answer_row.answer  # 답
            article_url = answer_row.article_url  # url
            article_title = answer_row.article_title  # title

            # URL 이용해서 엘라스틱서치로 본문 가져오기
            context = ReinforcementLearning.reinforce.get_content_by_URL(article_url)

            line = article_title + '§' + context + '§' + question + '§' + answer + '§' + str(
                context.find(answer)) + '\n'
            f.write(line)

        f.close()

        # txt -> json / 파일 경로 수정 필요
        f1score=ReinforcementLearning.reinforce.create_json(
            "C:/Users/DeepLearning_5/PycharmProjects/Play-Ball/ReinforcementLearning/ReinforcementFile/SelectedAnswerSet.txt",answers_no_list)
        context={"f1score":f1score}
        return HttpResponse(json.dumps(context),content_type="application/json")
