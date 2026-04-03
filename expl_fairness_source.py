


import os, random, numpy as np
import sys

# OUTPUT FILE RESULTS
sys.stdout = open("OUTPUT_OK.txt", "w", buffering=1)


SEED = 40


os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


'''
# ONE TARGET GENRE FOR EXPLORING FAIRNESS IN YELP
target_genres = [
    "Nightlife"
]
'''




#ML1M and HETREC. ONE TARGET GENRE AS EXAMPLE
target_genres = ["Drama",
                 ]




from scipy.stats import kendalltau

def calculate_jaccard(list_a, list_b):
    set_a, set_b = set(list_a), set(list_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0

def calculate_kendall(list_a, list_b):
    common_items = [item for item in list_a if item in list_b]
    if len(common_items) < 2:
        return 1.0

    idx_a = range(len(common_items))

    order_b = [list_b.index(item) for item in common_items]

    tau, _ = kendalltau(idx_a, order_b)
    return tau








# SIZE OF THE TOP K FAIRNESS-ORIENTED RECOMMENDATION LIST
K_all = {10,5}

# COEFFICIENT OF THE FAIRNESS MEASURE DPF
lambda_fairs=[0.01,0.1, 0.8]

for K in K_all:
    print("K=", K)
    if 1==1:

        if 1==1:



            import pandas as pd

            # 1. LOADING RATINGS
            ratings = pd.read_csv(
                "data/HETREC/ratings.dat",
                sep="::",
                engine="python",
                encoding="latin-1",
                names=["user_id","item_id","rating","timestamp"]
            )

            # 2. LOADING MOVIES
            movies = pd.read_csv(
                "data/HETREC/movies.dat",
                sep="::",
                engine="python",
                encoding="latin-1",
                names=["item_id","title","genres"]
            )

            # 3. LOADING ITEM GENRES/CATEGORIES
            movies["genres"] = movies["genres"].apply(lambda x: x.split("|") if isinstance(x,str) else ["Unknown"])
            item_groups = movies.explode("genres")[["item_id","genres"]].rename(columns={"genres":"group"})


            # 4. JOIN RATINGS WITH ITEM-GROUP
            data = ratings.merge(item_groups, on="item_id", how="left")
            data = data.merge(movies[["item_id","title"]], on="item_id", how="left")


            id_col = "item_id"

            item_genres = {
                row[id_col]: row["genres"].split("|") if isinstance(row["genres"], str) else row["genres"]
                for _, row in movies.iterrows()
            }


            all_genres = sorted({g for gs in item_genres.values() for g in gs})



            # 2. Baseline ranking

            from cornac.data import Dataset as CornacDataset
            from cornac.models import MF
            from cornac.eval_methods import RatioSplit
            import numpy as np
            from collections import defaultdict




            def evaluate_ndcg(topn_dict, cornac_dataset, K):
                import numpy as np
                from collections import defaultdict

                def get_user_ndcg(rec_list, ground_truth, k):
                    rec_list = rec_list[:k]
                    if not ground_truth:
                        return 0.0

                    dcg = 0.0
                    for i, item in enumerate(rec_list):
                        if item in ground_truth:
                            dcg += 1.0 / np.log2(i + 2)

                    num_relevant_in_topk = min(len(ground_truth), k)
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant_in_topk))

                    return dcg / idcg if idcg > 0 else 0.0

                ground_truth_dict = defaultdict(set)
                u_indices, i_indices, r_values = cornac_dataset.uir_tuple
                for idx in range(len(u_indices)):
                    if r_values[idx] > 0:
                        u_id = cornac_dataset.user_ids[u_indices[idx]]
                        i_id = cornac_dataset.item_ids[i_indices[idx]]
                        ground_truth_dict[u_id].add(i_id)

                all_ndcgs = []
                for u_id, pred_items in topn_dict.items():
                    if u_id in ground_truth_dict:
                        predicted_ids = [it[0] if isinstance(it, (tuple, list)) else it for it in pred_items]

                        user_score = get_user_ndcg(predicted_ids, ground_truth_dict[u_id], K)
                        all_ndcgs.append(user_score)

                return np.mean(all_ndcgs) if all_ndcgs else 0.0


            from cornac.metrics import NDCG


            def evaluate_ndcg_old(topn_dict, cornac_dataset, K):
                import numpy as np
                from collections import defaultdict

                def dcg(items, ground_truth, k):
                    items = items[:k]
                    score = 0.0
                    for i, item in enumerate(items):
                        if item in ground_truth:
                            score += 1.0 / np.log2(i + 2)
                    return score

                def ndcg(items, ground_truth, k):
                    actual_dcg = dcg(items, ground_truth, k)
                    num_hits = len(set(items[:k]) & set(ground_truth))
                    if num_hits == 0:
                        return 0.0
                    ideal_items = [1] * num_hits
                    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(num_hits))
                    return actual_dcg / ideal_dcg

                user_true_items = defaultdict(set)
                for entry in cornac_dataset.uir_tuple:
                    u_idx, i_idx, r = int(entry[0]), int(entry[1]), entry[2]
                    if r > 0:
                        user_true_items[cornac_dataset.user_ids[u_idx]].add(cornac_dataset.item_ids[i_idx])

                scores = []
                for user_id, pred_items in topn_dict.items():
                    if user_id not in user_true_items:
                        continue

                    actual_ground_truth = user_true_items[user_id]

                    predicted_ids = [it[0] if isinstance(it, (tuple, list)) else it for it in pred_items]

                    scores.append(ndcg(predicted_ids, actual_ground_truth, K))

                return np.mean(scores) if scores else 0.0




            vaecf_split = RatioSplit(
                data=ratings[["user_id", "item_id", "rating"]].values.tolist(),
                test_size=0.2,
                exclude_unknowns=True,
                seed=SEED
            )


            train_set = vaecf_split.train_set
            test_set = vaecf_split.test_set


            mf_model = MF(
                k=30,
                max_iter=50,
                learning_rate=0.01,
                lambda_reg=0.02,
                seed=SEED,
                verbose=True
            )
            mf_model.fit(vaecf_split.train_set)



            '''
            
            For VAECF use this model:
              from cornac.models import VAECF
            
              vaecf_model = VAECF(
                k=30,                         # reducido para velocidad
                autoencoder_structure=[20],   # más ligero
                act_fn="tanh",
                likelihood="mult",
                n_epochs=50,                  # menos épocas
                batch_size=512,               # batches más grandes
                learning_rate=0.001,
                beta=0.2,
                seed=SEED,
                verbose=True
            )
            
            
            '''

            #assert vaecf_model.is_trained


            N = 50

            topn_pred_train = defaultdict(dict)
            topn_pred_test = defaultdict(dict)

            users = list(train_set.uid_map.keys())
            rng = np.random.RandomState(SEED)
            rng.shuffle(users)

            cut = int(0.6 * len(users))
            users_train = set(users[:cut])
            users_test = set(users[cut:])

            num_items = train_set.num_items
            count_cornacusers=0
            for user_id, uidx in train_set.uid_map.items():
                count_cornacusers += 1

                user_scores = mf_model.score(uidx)

                top_indices = np.argsort(user_scores)[::-1][:N]

                items = [
                    (train_set.item_ids[iidx], float(user_scores[iidx]))
                    for iidx in top_indices
                ]

                if user_id in users_train:
                    topn_pred_train["VAECF"][user_id] = items
                elif user_id in users_test:
                    topn_pred_test["VAECF"][user_id] = items

            print(f"Train users: {len(topn_pred_train['VAECF'])}")
            print(f"Test users: {len(topn_pred_test['VAECF'])}")




            for target_genre in target_genres:
                print("---------GENRE----------"+target_genre)



                def compute_Rc(ratings_df, item_genres, target_genre):
                    total_interactions = len(ratings_df)

                    target_interactions = sum(
                        1
                        for iid in ratings_df["item_id"]
                        if target_genre in item_genres.get(iid, [])
                    )

                    return target_interactions / total_interactions

                Rc = compute_Rc(ratings, item_genres, target_genre)
                print(f"Rc ({target_genre}) = {Rc:.4f}")


                # 3. FAIRNESS-ORIENTED RE-RANKING
                import pulp

                def fair_topk_rerank_debug(
                        top_items,
                        item_genres,
                        K,
                        target_genre,
                        lambda_fair,
                        dpf_mode=0,
                        Rc=None
                ):

                    scores = [score for _, score in top_items]
                    s_min = min(scores)
                    s_max = max(scores)

                    if s_max - s_min > 1e-9:
                        top_items = [(iid, (s - s_min) / (s_max - s_min)) for iid, s in top_items]
                    else:
                        top_items = [(iid, 1.0) for iid, _ in top_items]


                    FAIR_SCALE = 1.0
                    import math

                    I1 = {i for i, _ in top_items if target_genre in item_genres.get(i, [])}
                    I2 = {i for i, _ in top_items if target_genre not in item_genres.get(i, [])}

                    if len(I1) == 0:
                        sel_items = [i for i, _ in top_items[:K]]
                        return sel_items, 0.0




                    pos = {i: idx for idx, (i, _) in enumerate(top_items)}


                    w = {i: 1.0 / math.log2(pos[i] + 2) for i, _ in top_items}


                    Z = sum(sorted(w.values(), reverse=True)[:K])



                    prob = pulp.LpProblem("FairTopK", pulp.LpMaximize)

                    x = {i: pulp.LpVariable(f"x_{str(i)}", cat="Binary")
                         for i, _ in top_items}

                    # --------------------------------------------------
                    # FAIRNESS TERM (switchable)
                    # --------------------------------------------------

                    if dpf_mode == 0:
                        # === ORIGINAL DPF (asymmetric, group vs non-group)
                        dpf_expr = (
                                pulp.lpSum([x[i] for i, _ in top_items if i in I1]) / K
                                -
                                pulp.lpSum([x[i] for i, _ in top_items if i in I2]) / K
                        )

                    elif dpf_mode == 1:
                        # === UNIFORM DISPARATE VISIBILITY
                        if Rc is None:
                            raise ValueError("Rc must be provided when dpf_mode == 1")

                        dpf_expr = (
                                pulp.lpSum([x[i] for i, _ in top_items if i in I1]) / K
                                - Rc
                        )

                    elif dpf_mode == 2:
                        # === POSITION-AWARE (NDCG-like) DPF
                        dpf_expr = (
                                pulp.lpSum([w[i] * x[i] for i, _ in top_items if i in I1]) / Z
                                -
                                pulp.lpSum([w[i] * x[i] for i, _ in top_items if i in I2]) / Z
                        )


                    else:
                        raise ValueError("Unknown dpf_mode")

                    # --------------------------------------------------
                    # OBJECTIVE
                    # --------------------------------------------------

                    prob += (
                            (1-lambda_fair)*pulp.lpSum([score * x[i] for i, score in top_items])/(K)
                            - lambda_fair * FAIR_SCALE * dpf_expr
                    )

                    # Top-K constraint
                    prob += pulp.lpSum([x[i] for i, _ in top_items]) == K

                    prob.solve(pulp.PULP_CBC_CMD(msg=False, threads=1))

                    sel_items = [i for i, _ in top_items if pulp.value(x[i]) == 1]

                    # Fairness value (post-hoc, same logic)
                    if dpf_mode == 0:
                        dpf_val = (
                                sum(1 for i in sel_items if i in I1) / K
                                -
                                sum(1 for i in sel_items if i in I2) / K
                        )
                    elif dpf_mode == 1:
                        dpf_val = (
                                sum(1 for i in sel_items if i in I1) / K
                                - Rc
                        )

                    elif dpf_mode == 2:
                        dpf_val = (
                                sum(w[i] for i in sel_items if i in I1) / Z
                                -
                                sum(w[i] for i in sel_items if i in I2) / Z
                        )



                    return sel_items, dpf_val

                from collections import defaultdict

                # NUMBER OF RECOMMENDATIONS IN THE INITIAL LIST
                N = 50

                reranked_train = defaultdict(dict)
                reranked_test  = defaultdict(dict)


                changes_train = 0
                total_users_train = 0


                changes_test = 0
                total_users_test = 0

                metrica_fairness = [1,0,2]




                for metrica in metrica_fairness:

                    print("metric=", metrica)
                    for lambda_fair in lambda_fairs:
                        changes_train = 0
                        total_users_train = 0
                        changes_test = 0
                        total_users_test = 0
                        print("lambda_fair=", lambda_fair)
                        print(f"Train users: {len(topn_pred_train['VAECF'])}")
                        print(f"Test users: {len(topn_pred_test['VAECF'])}")

                        jaccard_scores = []
                        kendall_scores = []

                        for model_name in ["VAECF"]:
                            for user, items in topn_pred_train["VAECF"].items():
                                if metrica==0:
                                    reranked_train["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica
                                    )
                                if metrica==1:
                                    reranked_train["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica,Rc
                                    )


                                if metrica==2:
                                    reranked_train["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica
                                    )



                                original = [i for i, _ in topn_pred_train["VAECF"][user][:K]]  # top-K original

                                sel_items, dpf_val = reranked_train["VAECF"][user]
                                reranked = sel_items

                                jaccard_scores.append(calculate_jaccard(original, reranked))
                                kendall_scores.append(calculate_kendall(original, reranked))

                                total_users_train += 1
                                if original != reranked:
                                    changes_train += 1
                            print(f"Jaccard Similarity average: {np.mean(jaccard_scores):.4f}")
                            print(f"Kendall's Tau average: {np.mean(kendall_scores):.4f}")



                            # --- EVALUATING NDCG FOR TRAIN ---
                            original_train_dict = {u: [it[0] for it in items[:K]] for u, items in topn_pred_train["VAECF"].items()}

                            reranked_train_dict = {u: items[0] for u, items in reranked_train["VAECF"].items()}


                            ndcg_orig_train = evaluate_ndcg(original_train_dict, vaecf_split.test_set, K)
                            ndcg_fair_train = evaluate_ndcg(reranked_train_dict, vaecf_split.test_set, K)
                            print(f"\n--- Results UTILITY (TRAIN) ---")
                            loss_train = ((ndcg_orig_train - ndcg_fair_train) / ndcg_orig_train) if ndcg_orig_train > 0 else 0
                            print(f"NDCG Original (Train): {ndcg_orig_train:.4f}")
                            print(f"NDCG Fair (Train): {ndcg_fair_train:.4f}")
                            print(f"Utility Loss Train: {loss_train:.2%}")



                            jaccard_scores = []
                            kendall_scores = []

                            for user, items in topn_pred_test["VAECF"].items():
                                if metrica==0:
                                    reranked_test["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica
                                    )
                                if metrica==1:
                                    reranked_test["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica,Rc
                                    )
                                if metrica==2:
                                    reranked_test["VAECF"][user] = fair_topk_rerank_debug(
                                        items, item_genres, K, target_genre, lambda_fair,metrica
                                    )


                                original = [i for i, _ in topn_pred_test["VAECF"][user][:K]]  # top-K original

                                sel_items, dpf_val = reranked_test["VAECF"][user]
                                reranked = sel_items

                                jaccard_scores.append(calculate_jaccard(original, reranked))
                                kendall_scores.append(calculate_kendall(original, reranked))



                                total_users_test += 1
                                if original != reranked:
                                    changes_test += 1

                            print(f"Jaccard Similarity average: {np.mean(jaccard_scores):.4f}")
                            print(f"Kendall's Tau average: {np.mean(kendall_scores):.4f}")


                            usuarios_con_test_real = set(topn_pred_test["VAECF"].keys()) & set(vaecf_split.test_set.uid_map.keys())


                            original_test_dict = {u: [it[0] for it in items[:K]] for u, items in topn_pred_test["VAECF"].items()}
                            reranked_test_dict = {u: items[0] for u, items in reranked_test["VAECF"].items()}


                            ndcg_orig = evaluate_ndcg(original_test_dict, vaecf_split.test_set, K)
                            ndcg_fair = evaluate_ndcg(reranked_test_dict, vaecf_split.test_set, K)

                            loss_test = ((ndcg_orig - ndcg_fair) / ndcg_orig) if ndcg_orig > 0 else 0
                            print(f"NDCG Original (Test): {ndcg_orig:.4f}")
                            print(f"NDCG Fair (Test): {ndcg_fair:.4f}")
                            print(f"Utility Loss Test: {loss_test:.2%}")


                            print(f"\nUsers with changes (train): {changes_train}/{total_users_train}")
                            print(f"Proportion: {changes_train/total_users_train:.2%}")

                            print(f"\nUsers with changes (test): {changes_test}/{total_users_test}")
                            print(f"Proportion: {changes_test/total_users_test:.2%}")



                        # ------------------------
                        # 4. ATTRIBUTE-LEVEL OUTCOME ANALYSIS
                        # ------------------------


                        import pandas as pd
                        from sklearn.tree import DecisionTreeClassifier, export_text

                        def build_rerank_dataset(topn_pred, reranked_topk, item_attributes, K, model_name="VAECF"):
                            data = []
                            for user in topn_pred[model_name]:
                                original_rank = [i for i, _ in topn_pred[model_name][user][:K]]
                                rerank_rank = reranked_topk[model_name][user]
                                rerank_rank1=rerank_rank[0]
                                for i, item_id in enumerate(original_rank):

                                    if item_id in rerank_rank1:
                                        delta = rerank_rank1.index(item_id) - i  # POSITION CHANGE
                                    else:
                                        delta = +K

                                    # LABEL DEFINITION: UP (+1), DOWN (-1)
                                    if delta < 0:
                                        label = 1
                                    elif delta > 0:
                                        label = -1
                                    else:
                                        label = 0
                                        continue

                                    attrs = item_attributes.get(item_id, {})
                                    row = {"item_id": item_id, "delta": delta, "label": label}
                                    row.update(attrs)
                                    data.append(row)

                            return pd.DataFrame(data)


                        # =======================
                        #  5. - Surrogate Model
                        # =======================


                        all_genres = sorted({g for gs in item_genres.values() for g in gs})

                        item_attributes = {}
                        for iid, genres in item_genres.items():
                            gs = genres if isinstance(genres, (list, tuple)) else [genres]
                            attrs = {f"__{g}": int(g in gs) for g in all_genres}
                            attrs["num_genres"] = len(gs)

                            item_attributes[iid] = attrs.copy()
                            item_attributes[str(iid)] = attrs.copy()

                        df_train = build_rerank_dataset(
                            topn_pred_train, reranked_train, item_attributes, K
                        )
                        if df_train.empty:
                            continue
                        df_train = df_train[df_train["label"].isin([1, -1])]



                        X_train = df_train.drop(
                            columns=["item_id", "delta", "label", "num_genres", "__"+target_genre]
                        )
                        y_train = df_train["label"]

                        clf = DecisionTreeClassifier(random_state=SEED, criterion="entropy", max_depth=3,min_samples_leaf=5, min_samples_split=4)
                        clf.fit(X_train, y_train)

                        tree_rules = export_text(clf, feature_names=list(X_train.columns))
                        print("\n=== Rules ===")
                        print(tree_rules)



                        import numpy as np
                        from sklearn.tree import _tree

                        def extract_class_rules(tree, feature_names, class_names, target_classes=[-1, 1]):
                            tree_ = tree.tree_
                            feature_name = [
                                feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
                                for i in tree_.feature
                            ]

                            paths = []

                            def recurse(node, path, paths):
                                if node == -1:
                                    return
                                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                                    name = feature_name[node]
                                    threshold = tree_.threshold[node]
                                    recurse(tree_.children_left[node],
                                            path + [f"({name} <= {threshold:.2f})"], paths)

                                    recurse(tree_.children_right[node],
                                            path + [f"({name} > {threshold:.2f})"], paths)
                                else:
                                    class_id = np.argmax(tree_.value[node])
                                    predicted_class = class_names[class_id]
                                    if predicted_class in target_classes:
                                        paths.append((path, predicted_class))

                            recurse(0, [], paths)
                            return paths


                        rules = extract_class_rules(clf, list(X_train.columns), clf.classes_, target_classes=[-1, 1])

                        print("\n=== Rules for items that promote (+1) or demote (-1) ===")
                        for conds, label in rules:
                            print(f"IF {' AND '.join(conds)} THEN label={label}")


                        from sklearn.metrics import accuracy_score

                        y_pred = clf.predict(X_train)
                        change_fidelity = accuracy_score(y_train, y_pred)

                        print(f"Change fidelity (items): {change_fidelity:.3f}")





                        df_test = build_rerank_dataset(
                            topn_pred_test, reranked_test, item_attributes, K
                        )

                        if df_test.empty:
                            continue

                        df_test = df_test[df_test["label"].isin([1, -1])]

                        X_test = df_test.drop(
                            columns=["item_id", "delta", "label", "num_genres", "__"+target_genre]
                        )
                        y_test = df_test["label"]


                        from sklearn.metrics import accuracy_score

                        y_pred_test = clf.predict(X_test)
                        fidelity_test = accuracy_score(y_test, y_pred_test)

                        print(f"Change fidelity TEST (out-of-sample): {fidelity_test:.3f}")
                        print(f"#Changes test: {len(y_test)}")
