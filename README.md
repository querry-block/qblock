# Querry Block

Querry Block is a simple and ligthweigth script to monitor and adaptively modify access structures to communication channels. At this moment we only support whatsapp, but we are currently working on expanding to other messaging apps, like telegram and signal.
When you read this, your are probably one of two people: either you have somebody that you want to block, but you dont like blocking possibly important information, or you got a message from our atomatic contacting system. In either case you migth want to know what we are doing: We offer a way, for you to set communication limits, meaning that after you set up our tool, you can specify a maximum number of messages per channel, that you can receive, and after this number is fulfilled, the channel gets blocked automatically. Why is this better than just blocking somebody yourself? The person youre anoiyed with knows that he can only write for example 12 remaining messages, so hopefully he behaves more responsible messaging you, and if not? than with every message you read, you know that youre one step closer to solving your problems.
# So how to use Querry Block?
First you require a simple linux shell, running some variant of ubuntu. Sample shells can be found by just using google. Then you need to clone this repository and run the python script called source/main.py on a seperate screen (so you can close the connection to the shell afterwards). It will allow you to connect you to your mobile (please use on of the guides you find after googling it for an explanation how to get the requested number), choose a contact (it is not yet possible to block groups) and a maximum number of messages this contact can send you. Querry Block will then periodically check your messages, and count the ones from this contact, until the number is reached. Then this person will automatically be blocked, but please note, that it can sometimes take some time to apply the block, this is done to save mobile data, since Querry Block needs a direct connection to your mobile.
