import argparse
from model import covid_model

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")
#added a new argument to check which model and configuration to train the dataset on
parser.add_argument("--mode")
#added a new argument named layers to get the layers wanted for the custom model
parser.add_argument("--layers",default=2)

args = parser.parse_args()

trainloader, testloader, validloader, train_data = load_data(args.data_dir)
#this is for the train model from scratch model in part 2
if (args.mode == "healthy"):
	model = healthy_model(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)
#this is for the resnetmodel in part 1
elif (args.mode == "covid"):
	model = covid_model(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)
if args.save_dir:
    save_checkpoint(model, args.save_dir)