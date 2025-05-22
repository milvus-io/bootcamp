class Config:
    # Define a class named Config to hold configuration settings.
    def __init__(self):
        # Initialize method to set default values for configuration settings.
        # self.download_path = "./images"
        self.download_path = "/home/xuyu/david.wxy/demos/cir_demo_with_milvus/images"
        # Set the path where images will be downloaded to "./images".
        self.imgs_per_category = 300
        # Define the number of images to download per category, set to 300.
        # self.milvus_uri = "milvus.db"
        self.milvus_uri = "http://localhost:19530"
        # Set the URI for the Milvus database, you can change to "http://localhost:19530" for a standard Milvus.
        self.collection_name = "demo_cir_demo"
        # Define the name of the collection in the Milvus database, set to "cir_demo_large".
        self.device = "cpu"
        # Set the device to use for computations, in this case, "gpu", you can change it to "cpu".
        self.model_type = "large"
        # Specify the type of model to use, set to "large".
        # self.model_path = "./models/magic_lens_clip_large.pkl"
        self.model_path = (
            "/home/xuyu/david.wxy/cir_demo2/models/magic_lens_clip_large.pkl"
        )

        # Set the path to the model file, default is "./magic_lens_clip_large.pkl".
