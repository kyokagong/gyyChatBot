
from django.http import JsonResponse

from django.shortcuts import render

from chatbot.src.python.dgk.dgk_util import test_main, sleep_task
from chatbot.src.python.action.controller import get_dgk_log

# Create your views here.


def home(request):
    return render(request, 'views/index.html')


def run_dgk_test(request):
    test_main()
    return JsonResponse({"code":1})

def get_dgk_logs(request):
    logs = get_dgk_log()
    return JsonResponse({"logs":logs})