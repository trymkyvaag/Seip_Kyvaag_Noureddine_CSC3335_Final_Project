from Convolution import Conv_Neur_Net

conv = Conv_Neur_Net()

conv.create_model((32, 3, 'relu'),
                  [(10, 'relu')],
                  )

print(conv.model.predict(conv.tweets_train))
print(conv.model.predict(conv.tweets_test))

pass