"""
URL configuration for fyp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from django.urls import path, include
from fyp import views
from fyp.views import predict_view, predict_view1, predict_view2, predict_view3,simple_chatbot,predict_live_view,chat_with_bot,threat_suggestion_from_llm,azure_llm_suggestion_view,chat_with_bot,chat_with_bots




urlpatterns = [
    path('', views.home),
    path('predict/', views.prediction),
    # path('pre',views.predict_view),
    path('pre/', predict_view),
    path('pre1/', predict_view1),
    path('pre2/', predict_view2),
    path('pre3/', predict_view3),
    # path('chat/stream/', chatbot_stream),
    path('chat/', simple_chatbot),
    path('predict-live/', predict_live_view),
    # path('api/predict-llm/', predict_llm_view, name='predict_llm_view'),
    path('api/predict-only/', views.predict_only),
    path('api/llm-suggest-only/', views.llm_suggestion_view),
    path('chatgpt/', views.chat_with_bot),
    path('chatgpt_suggession/', views.threat_suggestion_from_llm),
    path('api/azure-llm-suggest/', views.azure_llm_suggestion_view),
    path('api/chat_with_bot/', views.chat_with_bot),
    path('api/chat_with_bots/', views.chat_with_bots), 
]
