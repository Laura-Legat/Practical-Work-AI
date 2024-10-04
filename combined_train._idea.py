#GRU4Rec FIT

def fit(self, data, sample_cache_max_size=10000000, compatibility_mode=True, item_key='ItemId', session_key='SessionId', time_key='Time', combination=None, alpha=[0.2]):

        # Dataloader for gru4rec data
        self.data_iterator = SessionDataIterator(data, self.batch_size, n_sample=self.n_sample, sample_alpha=self.sample_alpha, sample_cache_max_size=sample_cache_max_size, item_key=item_key, session_key=session_key, time_key=time_key, session_order='time', device=self.device)

        # init ghru4rec model
        model = GRU4RecModel(self.data_iterator.n_items, self.layers, self.dropout_p_embed, self.dropout_p_hidden, self.embedding, self.constrained_embedding).to(self.device)
        self.model = model

        # init gru4rec optim
        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)


        # load ex2vec best params
        ex2vec_best_param_str = gru4rec_utils.convert_to_param_str('/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/optim/best_params_ex2vec.json')

        ex2vec_config = OrderedDict([x.split('=') for x in ex2vec_best_param_str.split(',') if "=" in x])

        # init new ex2vec
        config = config = {
                "alias": 'ex2vec_baseline_finaltrain_DEL',
                "num_epoch": int(ex2vec_config['num_epoch']),
                "batch_size": int(ex2vec_config['batch_size']),
                "optimizer": 'adam',
                "lr": float(ex2vec_config['learning_rate']),
                "rmsprop_alpha": float(ex2vec_config['rmsprop_alpha']),
                "momentum": float(ex2vec_config['momentum']),
                "n_users": 5, # change to 463 - check github
                "n_items": 146, # change to 879
                "latent_dim": 64,
                "num_negative": 0,
                "l2_regularization": float(ex2vec_config['l2_regularization']),
                "use_cuda": True,
                "device_id": 0,
                "pretrain": False,
                "pretrain_dir": '',
                "model_dir": "/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/models/{}_Epoch{}_f1{:.4f}.pt",
                "chckpt_dir":"/content/drive/MyDrive/JKU/practical_work/Practical-Work-AI/chckpts/{}_Epoch{}_f1{:.4f}.pt",
            }
        ex2vec = Ex2VecEngine(config)
        # set to train mode
        ex2vec.model.train()

        for epoch in range(self.n_epochs):
              for in_idx, out_idx, userids, sess_id, rel_ints, y in self.data_iterator(enable_neg_samples=(self.n_sample>0), reset_hook=reset_hook):
                for h in H: h.detach_()

                #SCORE GRU4Rec
                self.model.zero_grad() 
                R = self.model.forward(in_idx, H, out_idx, training=True) 

                # SCORE THE SAME DATA EX2VEC AND RETURN LOSS
                L_ex2vec = ex2vec.loss(torch.tensor(np.array(userids), device=self.device), torch.tensor(in_idx).cuda(), torch.tensor(np.array([np.pad(rel_int, (0, 50 - len(rel_int)), constant_values=-1) for rel_int in rel_ints]), device=self.device), torch.tensor(y).cuda())

                # CALC GRU4REC LOSS
                L_gru4rec = self.loss_function(R, out_idx, n_valid) / self.batch_size

                # COMBINE LOSSES
                L_combined = L_gru4rec + alpha * L_ex2vec

                #OPTIONAL
                # CALL FCT FOR UPDATE EX2VEC

                L_combined.backward() # GRU4REC GRADS
                opt.step() # UPDATE GRU4REC PARAMS
