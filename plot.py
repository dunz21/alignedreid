from util.colab import plot_pca

if __name__ == "__main__":
    folder_path = './people_2'
    model_path = "Alignedreid_models/Cuhk03_Resnet50_Alignedreid/checkpoint_ep300.pth.tar" #FUNCIONA
    # model_path = "Alignedreid_models/Market1501_Resnet50_Alignedreid(LS)/checkpoint_ep300.pth.tar" #FUNCIOAN
    plot_pca(folder_path=folder_path,simpleLegend=False,title='TEST XX',weight=model_path)
