CODE = """

    def train_network(X_train, X_test, y_train, y_test, network_type, layer_config):
        if network_type == 'feed_forward_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000)
        elif network_type == 'cascade_forward_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='logistic', solver='lbfgs', max_iter=1000)
        elif network_type == 'elman_backprop':
            hidden_layer_sizes = layer_config
            nn = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='tanh', solver='adam', max_iter=1000)

        nn.fit(X_train, y_train)

        y_pred_train = nn.predict(X_train)
        y_pred_test = nn.predict(X_test)

        r2_train = nn.score(X_train, y_train)
        r2_test = nn.score(X_test, y_test)

        mse_train = np.mean((y_pred_train - y_train) ** 2)
        mse_test = np.mean((y_pred_test - y_test) ** 2)

        return r2_train, r2_test, mse_train, mse_test, y_pred_test



    network_type = st.selectbox("Select network type",
                            ["feed_forward_backprop", "cascade_forward_backprop", "elman_backprop"])

    if network_type == "feed_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(10,), (20,)])
    elif network_type == "cascade_forward_backprop":
        layer_config = st.selectbox("Select layer configuration", [(20,), (10, 10)])
    elif network_type == "elman_backprop":
        layer_config = st.selectbox("Select layer configuration", [(15,), (5, 5, 5)])

    X = np.random.uniform(-5, 5, (1000, 2))
    y = target_func(X[:, 0], X[:, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    r2_train, r2_test, mse_train, mse_test, y_pred_test = train_network(X_train, X_test, y_train, y_test,
                                                                        network_type, layer_config)


    """