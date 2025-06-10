from django.urls import path

from DeepApp import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("LoadDataset", views.LoadDataset, name="LoadDataset"),
               path("FastText", views.FastText, name="FastText"),
               path("TrainML", views.TrainML, name="TrainML"),
               path("DetectFake.html", views.DetectFake, name="DetectFake"),
               path("DetectFakeAction", views.DetectFakeAction, name="DetectFakeAction"),
               path('Feedback/', views.FeedbackPage, name='FeedbackPage'),
               path('SubmitFeedback/', views.SubmitFeedback, name='SubmitFeedback'),
               ]


