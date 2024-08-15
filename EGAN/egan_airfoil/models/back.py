def metrics(epoch, generator, writer, *args, **kwargs):
    if (epoch + 1) % 100 == 0:
        generator.eval()

        def gen_func(latent, noise=None):
            if isinstance(latent, int):
                N = latent
                input = NoiseGenerator(N, cz, device=device)()
            else:
                N = latent.shape[0]
                if noise is None:
                    noise = np.zeros((N, cz[1]))
                input = torch.tensor(np.hstack([latent, noise]), device=device, dtype=torch.float)
            dp, cp, w, _, _ = generator(input)  # Adjust to capture cp, w
            return dp.cpu().detach().numpy().transpose([0, 2, 1]).squeeze(), cp, w

        X_test = np.load('../data/test.npy')
        X = np.load('../data/train.npy')

        _, cp, w = gen_func(cz[0])  # Use a sample latent input
        print(cp.shape, w.shape, _.shape)
        generator.save_cp_w(cp, w)  # Save control points and weights

        lsc = ci_cons(n_run, gen_func, cz[0])
        writer.add_scalar('Metric/LSC', lsc[0], epoch + 1)
        writer.add_scalar('Error/LSC', lsc[1], epoch + 1)

        rvod = ci_rsmth(n_run, gen_func, X_test)
        writer.add_scalar('Metric/RVOD', rvod[0], epoch + 1)
        writer.add_scalar('Error/RVOD', rvod[1], epoch + 1)

        div = ci_rdiv(n_run, X, gen_func)
        writer.add_scalar('Metric/Diversity', div[0], epoch + 1)
        writer.add_scalar('Error/Diversity', div[1], epoch + 1)

        mmd = ci_mmd(n_run, gen_func, X_test)
        writer.add_scalar('Metric/MMD', mmd[0], epoch + 1)
        writer.add_scalar('Error/MMD', mmd[1], epoch + 1)

        generator.train()


    def save_cp_w(self, cp, w, separator='|'):
        filename = 'cpw-NURBSx.json'
        self.save_count += 1


        try:
            cp_str = separator.join(map(str, cp.detach().cpu().tolist()))
            w_str = separator.join(map(str, w.detach().cpu().tolist()))

            save_tag = f"save_{self.save_count}"
            data_to_save = {save_tag: {'control_points': cp_str, 'weights': w_str}}

            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}  # If error, start a new file
                existing_data.update(data_to_save)
            else:
                existing_data = data_to_save

            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=4)
        except Exception as e:
            print(f"Failed to save JSON data: {e}")





    def forward(self, input):

        features = self.feature_generator(input)
        cp, w = self.cpw_generator(input)
        # print(cp.shape,w.shape)torch.Size([500, 2, 32]) torch.Size([500, 1, 32])
        self.save_cp_w(cp, w)
        # self.save_cp_w(cp, w)
        # print(cp,w)

        dp, ub, intvls = self.B(features, cp, w)
        # flops, params = profile(self.B, inputs=(features, cp, w))
        # print(f"FLOPs: {flops}")
        # print(f"Parameters: {params}")
        # FLOPs: 25327616.0
        # Parameters: 49087.0
        return dp, cp, w, ub, intvls

    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

    def save_cp_w(self, cp, w, separator='|'):
        if not self.training:  # Check if the model is in evaluation mode
            filename = 'cpw-Nx.json'
            self.save_count += 1

            try:
                cp_str = separator.join(map(str, cp.detach().cpu().tolist()))
                w_str = separator.join(map(str, w.detach().cpu().tolist()))

                save_tag = f"save_{self.save_count}"
                data_to_save = {save_tag: {'control_points': cp_str, 'weights': w_str}}

                if os.path.exists(filename):
                    with open(filename, 'r') as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            existing_data = {}  # If error, start a new file
                    existing_data.update(data_to_save)
                else:
                    existing_data = data_to_save

                with open(filename, 'w') as f:
                    json.dump(existing_data, f, indent=4)
            except Exception as e:
                print(f"Failed to save JSON data: {e}")
