'''
    Class for getting random responses.
    First val is representing the topic
    Second is a rand response
'''

from random import randint


class Response_generator():
    def __init__(self) -> None:
        self.responses = {}
        for i in range(8):
            self.responses[i] = []

        self.responses[0] = ['Did you really need to check this one? Looks good to us!',
                              'Very interesting topic! Tweet is good to post.',
                              'This is good to post.',
                              'Pretty standard tweet this, go ahead and post.',
                              ]
        self.responses[1] = ['You are better thant this...',
                             'This is not good to post, unless I misunderstood you?',
                             'Why would you say something like that?? Try again...',
                             'Still time to take this back, and NOT post it...'
                             ]
        self.responses[2] = ['That is not something to put on Twitter is it?',
                             'Would you have said this at a family dinner party?',
                             'What happened to thinking twice before posting online, or maybe that is why you came to me:)\nGood choice now do NOT post this',
                             'I do not know what you expect me to say... JUST NO',
                             'Holy moly, NO.']
        self.responses[3] = ['You have three seconds to take that back! Try something else',
                             'Not your best tweet... ',
                             'i am calling 911 on you. IP adress stored',
                             'How does......... Jail sound? New tweet thanks',
                             'I hate hate people like you, like why?',
                            'Immediate NO']
        
        self.responses[4] = ['Please, next....',
                            'Lets try not to be racist/sexcist next time?',
                            'https://www.goodtherapy.org/blog/dear-gt/why-do-i-troll-people-on-the-internet-how-can-i-stop',
                            'You need Jesus',
                            'Let us keep stuff like this to ourselves?']
        self.responses[5] = ['Good to post, love this',
                             'Good enery sensed. Post post post!',
                            'We need more people like u',
                            'Smiley face. That is all',
                            'You are the best, post this!', 
                            'yeyeyeyeyeyeeyey lovely']
        
        self.responses[6] = ['Why can you not accept people for who they are?',
                            'This is not it chief',
                            'Can you not think of something nice to post instead?',
                            'Come on now, next!']
        
        self.responses[7] = ['You know this is 2023 right?',
                             'I feel sorry for you',
                             'This better be a terrible joke',
                             'Why, just why? This is horrible',
                             'People like you are the problem']
        

    def get_response(self, topic_number): 
        r = randint(0,len(self.responses[topic_number]))
        return self.responses[topic_number][r]