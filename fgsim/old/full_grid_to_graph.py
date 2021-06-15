def grid_to_graph(caloimg, outformat="python"):
    caloimg = np.swapaxes(caloimg, 0, 2)
    nlayers, nrows, ncolumns = caloimg.shape

    # Node feature matrix
    feature_mtx_dyn = []
    feature_mtx_static = []
    # layer node
    layers = []

    # Save the edges
    edge_index = []
    # Keep track of connection per layer
    # layer node1 node2
    inner_edges_per_layer = [[] for _ in range(nlayers)]
    forward_edges_per_layer = [[] for _ in range(nlayers)]
    backward_edges_per_layer = [[] for _ in range(nlayers)]

    # save the type of connection 0 inlayer 1 forward 2 backward
    edge_attr = []

    # layer
    num_edges = 0
    globalid_to_indexD = {}
    runningindex = 0
    for ilayer in range(nlayers):
        # row
        for jrow in range(nrows):
            # column
            for kcolumn in range(ncolumns):
                if caloimg[ilayer, jrow, kcolumn] == 0:
                    continue
                # get the position in the faltened array
                curid = arrpos(ilayer, jrow, kcolumn, caloimg.shape)
                # save the position of the node
                globalid_to_indexD[curid] = runningindex
                # contruct the feature array for the node

                # extend the feature matrix
                features_dyn = np.array([caloimg[ilayer, jrow, kcolumn]])
                feature_mtx_dyn.append(features_dyn)

                features_static = np.array([float(ilayer), float(jrow), float(kcolumn)])
                feature_mtx_static.append(features_static)

                layers.append(ilayer)

                # regular neighbors
                # up if not top t
                if jrow != 0 and caloimg[ilayer, jrow - 1, kcolumn]:
                    targetid = arrpos(ilayer, jrow - 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)

                    num_edges = num_edges + 1
                # down
                if jrow != caloimg.shape[1] - 1 and caloimg[ilayer, jrow + 1, kcolumn]:
                    targetid = arrpos(ilayer, jrow + 1, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # left
                if kcolumn != 0 and caloimg[ilayer, jrow, kcolumn - 1]:
                    targetid = arrpos(ilayer, jrow, kcolumn - 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # right
                if (
                    kcolumn != caloimg.shape[1] - 1
                    and caloimg[ilayer, jrow, kcolumn + 1]
                ):
                    targetid = arrpos(ilayer, jrow, kcolumn + 1, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    inner_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(0)
                    num_edges = num_edges + 1
                # forward
                if (
                    ilayer != caloimg.shape[0] - 1
                    and caloimg[ilayer + 1, jrow, kcolumn]
                ):
                    targetid = arrpos(ilayer + 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    forward_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(1)
                    num_edges = num_edges + 1
                # backward
                if ilayer != 0 and caloimg[ilayer - 1, jrow, kcolumn]:
                    targetid = arrpos(ilayer - 1, jrow, kcolumn, caloimg.shape)
                    edge = np.array([curid, targetid])
                    edge_index.append(edge)
                    backward_edges_per_layer[ilayer].append(edge)
                    edge_attr.append(2)
                    num_edges = num_edges + 1
                runningindex = runningindex + 1
    # the DetIds in the adjacency matrix need to be
    # tranformed to real indices of the feature matrix
    for i in range(num_edges):
        edge_index[i][0] = globalid_to_indexD[edge_index[i][0]]
        edge_index[i][1] = globalid_to_indexD[edge_index[i][1]]

    # The following is not needed, because for python
    # the manipulated edge is the same for all these matrices:
    # edge_index[0] is inner_edges_per_layer[0][0] etc.
    # for ajdmtxi in (
    #     inner_edges_per_layer,
    #     forward_edges_per_layer,
    #     backward_edges_per_layer,
    # ):
    #     for ilayer in range(nlayers):
    #         for i in range(num_edges):
    #             ajdmtxi[ilayer][i][0] = globalid_to_indexD[ajdmtxi[ilayer][i][0]]
    #             ajdmtxi[ilayer][i][1] = globalid_to_indexD[ajdmtxi[ilayer][i][1]]
    if outformat == "np":
        return (
            np.array(feature_mtx_dyn, dtype=object),
            np.array(feature_mtx_static, dtype=object),
            np.array(edge_index, dtype=object),
            [np.array(e, dtype=object) for e in inner_edges_per_layer],
            [np.array(e, dtype=object) for e in forward_edges_per_layer],
            [np.array(e, dtype=object) for e in backward_edges_per_layer],
        )
    elif outformat == "ak":
        return ak.Array(
            {
                "feature_mtx_dyn": [feature_mtx_dyn],
                "edge_index": [edge_index],
                "inner_edges_per_layer": [inner_edges_per_layer],
                "forward_edges_per_layer": [forward_edges_per_layer],
                "backward_edges_per_layer": [backward_edges_per_layer],
            }
        )
    elif outformat == "geo":
        feature_mtx_dyn = torch.tensor(feature_mtx_dyn, dtype=torch.float32)
        feature_mtx_static = torch.tensor(feature_mtx_static, dtype=torch.float32)
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        inner_edges_per_layer = [
            torch.tensor(e, dtype=torch.int64).T for e in inner_edges_per_layer
        ]

        forward_edges_per_layer = [
            torch.tensor(e, dtype=torch.int64).T for e in forward_edges_per_layer
        ]

        backward_edges_per_layer = [
            torch.tensor(e, dtype=torch.int64).T for e in backward_edges_per_layer
        ]

        layers = torch.tensor(layers, dtype=torch.int64)

        graph = Data(x=feature_mtx_dyn, edge_index=edge_index.T)
        graph.feature_mtx_static = feature_mtx_static
        graph.layers = layers
        graph.inner_edges_per_layer = inner_edges_per_layer
        graph.forward_edges_per_layer = forward_edges_per_layer
        graph.backward_edges_per_layer = backward_edges_per_layer
        return graph.cpu()

    return (
        feature_mtx_dyn,
        edge_index,
        inner_edges_per_layer,
        forward_edges_per_layer,
        backward_edges_per_layer,
    )
