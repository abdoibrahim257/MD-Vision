import torch
import torch.nn as nn
import numpy as np
from torchvision.models import densenet121, resnet50, resnet152 , ResNet50_Weights, ResNet152_Weights, DenseNet121_Weights
from tqdm import tqdm
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

# Make a class that chooses the visual extractor based on the model name
class VisualExtractor(nn.Module):
    def __init__(self, model_name = 'resnet152', output_embed_size = 1024 , momentum = 0.1, pretrained = True):
        super(VisualExtractor, self).__init__()
        
        self.model_name = model_name
        self.output_embed_size = output_embed_size
        self.momentum = momentum
        self.pretrained = pretrained
        
        # Check the model name and initialize the corresponding visual extractor
        if model_name == 'densenet121':
            self.visual_extractor = DenseNet121_Visual_Extractor(self.output_embed_size, self.momentum, self.pretrained)
        elif model_name == 'resnet50':
            self.visual_extractor = ResNet50_Visual_Extractor(self.output_embed_size, self.momentum, self.pretrained)
        elif model_name == 'resnet152':
            self.visual_extractor = ResNet152_Visual_Extractor(self.output_embed_size, self.momentum, self.pretrained)
        elif model_name == 'hog_pca':
            self.visual_extractor = HOG_PCA_Visual_Extractor(self.output_embed_size, self.pretrained)
        else:
            raise ValueError(f"Unknown model name: {model_name}, Please choose from 'densenet121', 'resnet50', 'resnet152', 'hog_pca'")
        
    def forward(self, images):
        # Extract features from the images using the visual extractor model ( Features , Avg_features)
        return self.visual_extractor(images)

class DenseNet121_Visual_Extractor(nn.Module):
    def __init__(self, output_embed_size, momentum, pretrained):
            super(DenseNet121_Visual_Extractor, self).__init__()  # Call the init method of the parent class
            
            # Build the DenseNet-121 network from scratch with pretrained weights
            self.densenet = densenet121(weights = DenseNet121_Weights.IMAGENET1K_V1)
            
            # Get the number of input features to the linear layer
            num_inputs = self.densenet.classifier.in_features
            
            # Check if the model is pretrained
            if pretrained:
                for param in self.densenet.parameters():
                    param.requires_grad = False
                    
            # Replace the final layer with a linear layer
            self.densenet.classifier = nn.Linear(num_inputs, output_embed_size)
            
            # Initialize the weights of the linear layer
            nn.init.kaiming_normal_(self.densenet.classifier.weight)
            nn.init.constant_(self.densenet.classifier.bias, 0)
            
            # Define a batch normalization layer for the output features
            self.bn = nn.BatchNorm1d(output_embed_size, momentum= momentum)
        
        
    def forward(self, images):
        """Forward pass of the DenseNet-121 model to extract features from the images.

        Args:
            images (torch.Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            visual_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the extracted features.
        """
        
        # Extract features from the images using the DenseNet-121 model
        features = self.densenet(images)
        
        # Pass the features through the batch normalization layer
        features = self.bn(features)
        
        return features

class ResNet50_Visual_Extractor(nn.Module):
    def __init__(self, output_embed_size, momentum , pretrained = True):
        super(ResNet50_Visual_Extractor, self).__init__()
        
        # Build the ResNet-50 network from scratch
        self.resnet = resnet50(weights = ResNet50_Weights.IMAGENET1K_V1)
        
        num_inputs = self.resnet.fc.in_features
        
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.resnet.fc = nn.Linear(num_inputs, output_embed_size)      # The final layer is called fc in the class
        
        # Initialize the weights of the linear layer
        nn.init.kaiming_normal_(self.resnet.fc.weight) # Using the kaiming normal initialization avoid the vanishing gradient problem
        nn.init.constant_(self.resnet.fc.bias, 0) # Initialize the bias to zero
        
        # Add a batch normalization layer
        self.bn = nn.BatchNorm1d(output_embed_size, momentum = momentum) # Using a small momentum value helps in faster convergence of the model
        
        
    def forward(self, images):
        """Forward pass of the ResNet-50 model to extract features from the images.

        Args:
            images (torch.Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            visual_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the extracted features.
            avg_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the average features.
        """
        # Extract features from the images
        features = self.resnet(images)
        
        # Pass the features through the batch normalization layer
        features = self.bn(features)

        return features


class ResNet152_Visual_Extractor(nn.Module):
    def __init__(self, output_embed_size, momentum, pretrained):
        super(ResNet152_Visual_Extractor, self).__init__()
        
        # Build the ResNet-152 network from scratch
        self.resnet = resnet152(weights = ResNet152_Weights.IMAGENET1K_V1)
        
        num_inputs = self.resnet.fc.in_features
        
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.resnet.fc = nn.Linear(num_inputs, output_embed_size)      # The final layer is called fc in the class
        
        # Initialize the weights of the linear layer
        nn.init.kaiming_normal_(self.resnet.fc.weight) # Using the kaiming normal initialization avoid the vanishing gradient problem
        nn.init.constant_(self.resnet.fc.bias, 0) # Initialize the bias to zero
        
        # Add a batch normalization layer
        self.bn = nn.BatchNorm1d(output_embed_size, momentum = momentum) # Using a small momentum value helps in faster convergence of the model
        
        
    def forward(self, images):
        """Forward pass of the ResNet-152 model to extract features from the images.

        Args:
            images (torch.Tensor): A batch of images of shape (N, C, H, W).

        Returns:
            visual_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the extracted features.
            avg_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the average features.
        """
        # Extract features from the images
        features = self.resnet(images)
        
        # Pass the features through the batch normalization layer
        features = self.bn(features)

        return features
    
class HOG_PCA_Visual_Extractor(nn.Module):
    def __init__(self, output_embed_size , pretrained):
        super(HOG_PCA_Visual_Extractor, self).__init__()
        
        # HOG parameters
        self.orientation = 10
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (4, 4)
        
        # PCA parameters
        self.pca = None
        self.n_components = output_embed_size
        
        # Check if the model is pretrained
        self.pretrained = pretrained
        
        if self.pretrained:
            self.pca = joblib.load('./Data/models/pca.pkl')
        else:
            self.pca = make_pipeline(StandardScaler(), PCA(n_components=self.n_components))
            
    def forward(self, images):
        """Forward pass of the HOG-PCA model to extract features from the images.
        
        Args:
            images (torch.Tensor): A batch of images of shape (N, C, H, W).
            
        Returns:
            visual_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the extracted features.
            avg_features (torch.Tensor): A tensor of shape (N, output_embed_size) containing the average features.
        """
        
        # Extract features from the images using the HOG-PCA model
        features = []
        print ("Extracting HOG features from the images")
        for image in tqdm(images):
            # Compute the HOG features
            hog_features = hog(image, orientations=self.orientation, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block)
            
            # Append the HOG features to the list of features
            features.append(hog_features)
        
        # Convert the list of features to a numpy array
        features = np.array(features)
        
        # Perform PCA on the features
        print ("Performing PCA on the HOG features")
        if self.pretrained:
            features = self.pca.transform(features)
        else:
            features = self.pca.fit_transform(features)
            # Save the PCA model
            joblib.dump(self.pca, './Data/models/pca.pkl')

        # Convert the features to a tensor
        features = torch.tensor(features)
        
        return features
    
class MLC(nn.Module):
    def __init__(self, vis_features_dim = 1024 , embed_dim = 512 , classes = 210 , k = 10 ):
        super(MLC, self).__init__()
        self.classifier = nn.Linear(vis_features_dim, classes)
        self.emdedding = nn.Embedding(classes, embed_dim)
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.__init_weights()
        
    def __init_weights(self):
        # Initialize the weights of the classifier with Kaiming normal initialization
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        # Initialize the weights of the embedding layer with Kaiming normal initialization
        nn.init.kaiming_normal_(self.emdedding.weight)
        
    def forward (self, visual_features):
        # Pass the visual features through the classifier
        tags = self.classifier(visual_features)
        
        # Apply the sigmoid activation function to the tags
        tags = self.sigmoid(tags)
        
        # Get the top k classes to pass through the embedding layer
        top_k_classes = torch.topk(tags, self.k, dim=1).indices
        
        # Get the embeddings for the top k classes
        embeddings = self.emdedding(top_k_classes)
        
        # Sum across the dim=1
        embeddings = torch.sum(embeddings, dim=1)
        
        # return tags, embeddings
        return top_k_classes, embeddings
        
    
class AttentionVisual(nn.Module):
    def __init__(self, vis_features_dim = 1024 , hidden_dim = 512 , att_dim = 256):
        super(AttentionVisual, self).__init__()
        
        # Visual Attention
        self.W_v = nn.Linear(vis_features_dim, att_dim)
        self.bn_v = nn.BatchNorm1d(1)
        
        # Decoder Attention
        self.W_v_h = nn.Linear(hidden_dim, att_dim)
        self.bn_v_h = nn.BatchNorm1d(att_dim)
        
        # Attention Layer
        self.W_v_att = nn.Linear(att_dim, 1)
        self.bn_v_att = nn.BatchNorm1d(1)
        
        # Activation Functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, vis_enc_output, dec_hidden_state):
            # This results in a tensor of shape (batch_size, num_pixels, att_dim)
            W_v = self.W_v(vis_enc_output)

            # This results in a tensor of shape (batch_size, att_dim)
            W_v_h = self.W_v_h(dec_hidden_state)

            # Apply ReLU activation function on the result
            # This results in a tensor of shape (batch_size, num_pixels, att_dim)
            join_output = self.relu(W_v + W_v_h.unsqueeze(1))

            # This results in a tensor of shape (batch_size, num_pixels)
            join_output = self.W_v_att(join_output).squeeze(2) 

            # Apply softmax function on the joined output to get attention scores
            att_scores = self.softmax(join_output) 

            # Multiply the attention scores with the visual encoder output and sum over the pixel dimension
            # This results in a tensor of shape (batch_size, feature_dim), representing the attended visual features
            att_output = torch.sum(att_scores.unsqueeze(2) * vis_enc_output, dim = 1)

            # Return the attended visual features and the attention scores
            return att_output

class AttentionSemantic(nn.Module):
    def __init__(self, sem_enc_dim = 512 , hidden_dim = 512 , att_dim = 256):
        super(AttentionSemantic, self).__init__()
        # Semantic Attention
        self.W_a = nn.Linear(sem_enc_dim, att_dim)
        # self.bn_a = nn.BatchNorm1d(att_dim)
        
        # Decoder Attention
        self.W_a_h = nn.Linear(hidden_dim, att_dim)
        # self.bn_a_h = nn.BatchNorm1d(att_dim)
        
        # Attention Layer
        self.W_a_att = nn.Linear(att_dim, 1)
        # self.bn_a_att = nn.BatchNorm1d(1)
        
        # Activation Functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        

    def forward(self, sem_enc_output, dec_hidden_state):
        # This results in a tensor of shape (batch_size, num_sentences, att_dim)
        W_a = self.W_a(sem_enc_output)
        
        # This results in a tensor of shape (batch_size, att_dim)
        W_a_h = self.W_a_h(dec_hidden_state)

        # Apply ReLU activation function on the result
        # This results in a tensor of shape (batch_size, num_sentences, att_dim)
        join_output = self.relu(W_a + W_a_h.unsqueeze(1))

        # This results in a tensor of shape (batch_size, num_sentences)
        join_output = self.W_a_att(join_output).squeeze(2)

        # Apply softmax function on the joined output to get attention scores
        att_scores = self.softmax(join_output)

        # Multiply the attention scores with the semantic encoder output and sum over the sentence dimension
        # This results in a tensor of shape (batch_size, feature_dim), representing the attended semantic features
        att_output = torch.sum(att_scores.unsqueeze(2) * sem_enc_output, dim = 1)

        return att_output


class SentenceLSTM(nn.Module):
    def __init__(self, vis_features_dim=1024, sem_features_dim=512, hidden_dim=512, att_dim=256, sent_input_dim=1024, word_input_dim=512, stop_dim=256, device='cuda:0'):    
        super(SentenceLSTM, self).__init__()

        # Define the dimensions of the visual and semantic features
        self.hidden_dim = hidden_dim
        self.word_input_dim = word_input_dim
    
        # Define the attention modules
        self.vis_att = AttentionVisual(vis_features_dim, hidden_dim, att_dim).to(device)
        self.sem_att = AttentionSemantic(sem_features_dim, hidden_dim, att_dim).to(device)

        # Define the context layer
        self.contextLayer = nn.Linear(vis_features_dim + sem_features_dim, sent_input_dim)
        # self.contextLayer = nn.Linear(vis_features_dim, sent_input_dim)
        # self.bn_context = nn.BatchNorm1d(sent_input_dim)
        
        # Define the LSTM cell
        self.lstm = nn.LSTMCell(sent_input_dim, hidden_dim, bias = True)
        
        # Define the topic 
        self.topic_hid_layer = nn.Linear(hidden_dim, word_input_dim)
        self.bn_topic = nn.BatchNorm1d(word_input_dim)
        self.topic_context_layer = nn.Linear(sent_input_dim, word_input_dim)
        # self.bn_topic_context = nn.BatchNorm1d(word_input_dim)
        
        # Define the stop vector
        self.stop_prev_hid = nn.Linear(hidden_dim, stop_dim)
        # self.bn_stop_prev_hid = nn.BatchNorm1d(stop_dim)
        
        self.stop_cur_hid = nn.Linear(hidden_dim, stop_dim)
        # self.bn_stop_cur_hid = nn.BatchNorm1d(stop_dim)
        
        # Define the final stop layer
        self.final_stop_layer = nn.Linear(stop_dim, 2)
        # self.bn_stop_layer = nn.BatchNorm1d(2)

        # Define the activation functions
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, vis_enc_output, sem_enc_output, s_max, device):
        """
        Forward propagation.

        :param vis_enc_output: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param captions: captions, a tensor of dimension (batch_size, max_no_of_sent, max_sent_len)
        :return: topic vector for word LSTM (batch_size, max_no_of_sent, word_input_dim), stop vector for each time step (batch_size, max_no_of_sent, 2)
        """
        batch_size = vis_enc_output.size(0)
        vis_features_dim = vis_enc_output.size(1)
        sem_features_dim = sem_enc_output.size(1)

        # Reshape the visual and semantic encoder outputs
        vis_enc_output = vis_enc_output.reshape(batch_size, -1, vis_features_dim)  # (batch_size, num_pixels, vis_features_dim)
        sem_enc_output = sem_enc_output.reshape(batch_size, -1, sem_features_dim)

        # Initialize the hidden and cell states for the LSTM
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Initialize the topic and stop vectors
        topics = torch.zeros(batch_size, s_max, self.word_input_dim, device=device)
        ps = torch.zeros(batch_size, s_max, 2, device=device)

        for t in range(s_max):
            # Attention Mechanisms
            # --------------------
            # Get the visual attention output
            vis_att_output = self.vis_att(vis_enc_output, h)  # (batch_size, vis_features_dim)
            
            # Get the semantic attention output
            sem_att_output = self.sem_att(sem_enc_output, h)
        
            # Combine the outputs of visual and semantic attention
            summed = torch.cat((vis_att_output, sem_att_output), dim=1)  # (batch_size, vis_features_dim + sem_features_dim)
            
            # Context Vector
            # --------------
            # Generate the context vector from the combined attention output
            context_output = self.contextLayer(summed)  # (batch_size, sent_input_dim)
        
            # LSTM Cell
            # ---------
            # Save the current hidden state for later use
            h_prev = h.clone()
        
            # Update the hidden state and cell state using the LSTM cell
            h, c = self.lstm(context_output, (h, c))  # (batch_size, hidden_dim), (batch_size, hidden_dim)
        
            # Topic Vector
            # ------------
            # Generate the topic vector from the previous hidden state and the context vector
            topic = self.topic_hid_layer(h_prev) + self.topic_context_layer(context_output)  # (batch_size, word_input_dim)
            topic = self.relu(topic)
        
            # Stop Vector
            # -----------
            # Generate the stop vector from the previous and current hidden states
            p = self.stop_prev_hid(h_prev) + self.stop_cur_hid(h)  # (batch_size, stop_dim)
            p = self.tanh(p)
            p = self.final_stop_layer(p)  # (batch_size, 2)
        
            # Store the generated topic vector and stop vector for this timestep
            topics[:, t, :] = topic
            ps[:, t, :] = p

        return topics, ps

class WordLSTM(nn.Module):
    def __init__(self, word_input_dim=512, word_hidden_dim=512, vocab_size= 2019, num_layers=1):
        super(WordLSTM, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, word_input_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(word_input_dim, word_hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(word_hidden_dim, vocab_size)
        
        self.word_input_dim = word_input_dim
        self.word_hidden_dim = word_hidden_dim
        self.vocab_size = vocab_size
        
        
        
    def forward(self, topic, caption):
        """
        Forward propagation.

        :param topic: topic vector, a tensor of dimension (batch_size, word_input_dim)
        :param caption: a token 
        :return: outputs predicting the next word, a tensor of dimension (batch_size, max_sent_len, vocab_size)
        """
        # Embed the caption
        # caption shape (batch_size, max_word_len)
        embeddings = self.embedding(caption)  # (batch_size, max_sent_len, word_input_dim)

        # Concatenate the topic and the embeddings along the second dimension
        # topic shape (batch_size, 1, word_input_dim)
        # embeddings shape (batch_size, max_sent_len, word_input_dim)
        lstm_input = torch.cat([topic.unsqueeze(1), embeddings], dim=1)  # (batch_size, max_sent_len + 1, word_input_dim)

        # Pass the input through the LSTM
        # outputs shape (batch_size, max_sent_len + 1, word_hidden_dim)
        outputs, _ = self.lstm(lstm_input)  # (batch_size, max_sent_len + 1, word_hidden_dim)

        # Pass the LSTM outputs through the fully connected layer
        outputs = self.fc(outputs)  # (batch_size, max_sent_len + 1, vocab_size)

        # Remove the last output along the second dimension
        outputs = outputs[:, :-1, :]  # (batch_size, max_sent_len, vocab_size)

        return outputs
    
    def forward_test (self , topic , input , n_max_words =40):
        """
        Forward propagation for the test phase.
        
        :param topic: topic vector, topic per sentence, a tensor of dimension (batch_size, 1, word_input_dim)
        :param n_max_words: maximum number of words to generate
        :return: outputs predicting the next word, a tensor of dimension (batch_size, vocab_size)
        """
        # Get the batch size and the features dimension
        batch_size = topic.size(0)

        # Initialize the outputs tensor
        outputs = torch.zeros(batch_size, n_max_words).to(topic.device)
        
        outputs[:, 0] = input[:, 0]
        # outputs[:, 1] = input[:, 1]
        
        # Initialize the input tensor
        input = torch.as_tensor(input, dtype=torch.long, device=topic.device)
        
        # Loop over the maximum number of words
        for i in range(1, n_max_words - 1):
            # Embed the input
            embeddings = self.embedding(input).to(topic.device)
            
            # Concatenate the topic and the embeddings
            lstm_input = torch.cat([topic.unsqueeze(1), embeddings], dim=1)
            
            # Pass the input through the LSTM
            h, _ = self.lstm(lstm_input)
            
            # Set h to the last hidden state
            h = h[:, -1, :]
            
            # Pass the LSTM outputs through the fully connected layer
            outputs1 = self.fc(h) # outputs shape (batch_size, 2, vocab_size)
            
            # Get the word with the maximum probability with argmax
            max_word_indices = torch.argmax(outputs1, dim=1).unsqueeze(1)
                        # Get the second word with the maximum probability with argmax when it predict the unknown token
            unk_index = 3
            for j in range(batch_size):
                if max_word_indices[j] == unk_index:
                    max_word_indices[j] = torch.argsort(outputs1[j], descending=True)[1].unsqueeze(0)
                    
            input = torch.cat((input, max_word_indices), dim=1) 
            
            # Update the outputs tensor
            outputs[:, i] = max_word_indices.squeeze(1)
            
        return outputs
    
    
# if __name__ == '__main__':
#     import warnings
#     warnings.filterwarnings("ignore")
#     images = torch.randn((4, 3, 224, 224))
#     captions = torch.ones((4, 10)).long()
#     hidden_state = torch.randn((4, 1, 512))

#     print("images:{}".format(images.shape))
#     print("captions:{}".format(captions.shape))
#     print("hidden_states:{}".format(hidden_state.shape))
 
#     extractor = VisualExtractor()
#     visual_features= extractor.forward(images)
#     print("visual_features:{}".format(visual_features.shape))

#     mlc = MLC()
#     tags, semantic_features = mlc.forward(visual_features)
#     print("tags:{}".format(tags.shape))
#     print("semantic_features:{}".format(semantic_features.shape))

#     co_att = CoAttention()
#     ctx= co_att.forward(visual_features, semantic_features, hidden_state)
#     print("ctx:{}".format(ctx.shape))

#     sent_lstm = SentenceLSTM()
#     topic, p_stop, hidden_state, states = sent_lstm.forward(ctx, hidden_state)
#     print("Topic:{}".format(topic.shape))
#     print("P_STOP:{}".format(p_stop.shape))

#     word_lstm = WordLSTM( )
#     words = word_lstm.forward(topic, captions)
#     print("words:{}".format(words.shape))

#     # Expected Output
#     # images: torch.Size([4, 3, 224, 224])
#     # captions: torch.Size([4, 1, 10])
#     # hidden_states: torch.Size([4, 1, 512])
#     # visual_features: torch.Size([4, 2048, 7, 7])
#     # tags: torch.Size([4, 156])
#     # semantic_features: torch.Size([4, 10, 512])
#     # ctx: torch.Size([4, 512])
#     # Topic: torch.Size([4, 1, 512])
#     # P_STOP: torch.Size([4, 1, 2])
#     # words: torch.Size([4, 1000])

#     # images = torch.randn((4, 3, 224, 224))
#     # captions = torch.ones((4, 3, 10)).long()
#     # prev_outputs = torch.randn((4, 512))
#     # now_words = torch.ones((4, 1))
#     #
#     # ctx_records = torch.zeros((4, 10, 512))
#     # captions = torch.zeros((4, 10)).long()
#     #
#     # print("images:{}".format(images.shape))
#     # print("captions:{}".format(captions.shape))
#     # print("hidden_states:{}".format(prev_outputs.shape))
#     #
#     # extractor = VisualFeatureExtractor()
#     # visual_features = extractor.forward(images)
#     # print("visual_features:{}".format(visual_features.shape))
#     #
#     # mlc = MLC()
#     # tags, semantic_features = mlc.forward(visual_features)
#     # print("tags:{}".format(tags.shape))
#     # print("semantic_features:{}".format(semantic_features.shape))
#     #
#     # co_att = CoAttention()
#     # ctx = co_att.forward(visual_features, semantic_features, prev_outputs)
#     # print("ctx:{}".format(ctx.shape))
#     #
#     # ctx_records[:, 0, :] = ctx
#     #
#     # sent_tcn = SentenceTCN()
#     # topic, p_stop, prev_outputs = sent_tcn.forward(ctx_records, prev_outputs)
#     # print("Topic:{}".format(topic.shape))
#     # print("P_STOP:{}".format(p_stop.shape))
#     # print("Prev_Outputs:{}".format(prev_outputs.shape))
#     #
#     # captions[:, 0] = now_words.view(-1,)
#     #
#     # word_tcn = WordTCN()
#     # words = word_tcn.forward(topic, captions)
#     # print("words:{}".format(words.shape))

    
