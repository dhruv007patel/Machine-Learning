Below mentioned are the modification done to the original code.

1. 	Code added: To create testset and testloader
	
	testset = datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

2. Code added: To added to tune hyper-parameters and L2 regularizer by setting weight_decay

	# Code added : To Tune hyper-parameters
	_num_epoch = 42
	_lr = 0.001
	_momentum = 0.9
	_weight_decay = 1e-4

	# Code added to update learing rate to decay lr after each epoch
	def update_lr(optimizer, lr):    
	    for param_group in optimizer.param_groups:
	        param_group['lr'] = lr

	 if (epoch+1) % 20 == 0:
          _lr /= 3
          update_lr(optimizer, _lr)

	optimizer = optim.SGD(list(model.fc.parameters()), lr = _lr, momentum = _momentum,weight_decay = _weight_decay)

3. Code added: To calculate the accuracy rate on test error and to save best model

	# Code added : To calculate the accuracy rate on test error and to save best model
    print('Testing images for : '+ str(epoch + 1) +" epoch(s)")
    best_acc = 0 
    correct = 0
    total = 0
    with torch.no_grad():
      for i,data in enumerate(testloader, 0):
        images, labels = data
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted.cuda() == labels.cuda()).sum().item()
    print('Accuracy :'+ str((100 * correct / total)))
    if(best_acc < (100 * correct / total)):
      best_acc = (100 * correct / total)
      torch.save(model,'model_best.pth')
      print("New Best Model Saved")

