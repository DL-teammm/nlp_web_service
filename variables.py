import gensim
import pymorphy2

import nltk
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')


sw = stopwords.words('russian')
additional_sw = 'мои оно мной мною мог могут мор мое мочь оба нам нами ними однако нему никуда наш нею неё наша наше ' \
                'наши очень отсюда вон вами ваш ваша ваше ваши весь всем всеми вся ими ею будем будете будешь буду ' \
                'будь будут кому кого которой которого которая которые который которых кем каждое каждая каждые ' \
                'каждый кажется та те тому собой тобой собою тобою тою хотеть хочешь свое свои твой своей своего ' \
                'своих твоя твоё сама сами теми само самом самому самой самого самим самими самих саму чему тебе ' \
                'такое такие также такая сих тех ту эта это этому туда этим этими этих абы аж ан благо буде вроде ' \
                'дабы едва ежели затем зато ибо итак кабы коли коль либо лишь нежели пока покамест покуда поскольку ' \
                'притом причем пускай пусть ровно сиречь словно также точно хотя чисто якобы '
pronouns = 'я мы ты вы он она оно они себя мой твой ваш наш свой его ее их то это тот этот такой таков столько весь ' \
           'всякий сам самый каждый любой иной другой кто что какой каков чей сколько никто ничто некого нечего ' \
           'никакой ничей нисколько кто-то кое-кто кто-нибудь кто-либо что-то кое-что что-нибудь что-либо какой-то ' \
           'какой-либо какой-нибудь некто нечто некоторый некий '
conjunctions = 'что чтобы как когда ибо пока будто словно если потому что оттого что так как так что лишь только как ' \
               'будто с тех пор как в связи с тем что для того чтобы кто как когда который какой где куда откуда '
digits = 'ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать тринадцать ' \
         'четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать двадцать тридцать сорок пятьдесят ' \
         'шестьдесят семьдесят восемьдесят девяносто сто '
modal_words = 'вероятно возможно видимо по-видимому кажется наверное безусловно верно  действительно конечно ' \
              'несомненно разумеется '
particles = 'да так точно ну да не ни неужели ли разве а что ли что за то-то как ну и ведь даже еще ведь уже все ' \
            'все-таки просто прямо вон это вот как словно будто точно как будто вроде как бы именно как раз подлинно ' \
            'ровно лишь только хоть всего исключительно вряд ли едва ли '
prepositions = 'близ  вблизи  вдоль  вокруг  впереди  внутрь  внутри  возле  около  поверх  сверху  сверх  позади  ' \
               'сзади  сквозь  среди  прежде  мимо  вслед  согласно  подобно  навстречу  против  напротив  вопреки  ' \
               'после  кроме  вместе  вдали  наряду  совместно  согласно  нежели вроде от бишь до без аж тех раньше ' \
               'совсем только итак например из прямо ли следствие а поскольку благо пускай благодаря случае затем ' \
               'притом также связи время при чтоб просто того невзирая даром вместо точно покуда тогда зато ради ан ' \
               'буде прежде насчет раз причине тому так даже исходя коль кабы более ровно либо помимо как-то будто ' \
               'если словно лишь бы и не будь пор тоже разве чуть как хотя наряду потому пусть в равно между сверх ' \
               'ибо на судя то чтобы относительно или счет за но сравнению причем оттого есть когда уж ввиду тем для ' \
               'дабы чем хоть с вплоть скоро едва после той да вопреки ежели кроме сиречь же коли под абы несмотря ' \
               'все пока покамест паче прямо-таки перед что по вдруг якобы подобно '
evaluative = 'наиболее наименее лучший больший высший низший худший более менее'

sw.extend(additional_sw.split())
sw.extend(pronouns.split())
sw.extend(conjunctions.split())
sw.extend(digits.split())
sw.extend(modal_words.split())
sw.extend(particles.split())
sw.extend(prepositions.split())
sw.extend(evaluative.split())
sw = list(set(sw))


analyzer = pymorphy2.MorphAnalyzer()
lemmatized_sw = [analyzer.parse(word)[0].normal_form for word in sw]

fasttext = gensim.models.fasttext.load_facebook_model('./models/cc.ru.300.bin')