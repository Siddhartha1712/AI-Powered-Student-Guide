class Treenode:
    def __init__(self, data):
        self.data = data
        self.child = []
        self.parent = None

    def add_child_node(self, child):
        child.parent = self
        self.child.append(child)

    def find_node(self, value):
        """
        Recursively search for a node with the given value.
        """
        if self.data == value:
            return self
        for child in self.child:
            result = child.find_node(value)
            if result:
                return result
        return None

    def print_tree(self, prefix="", is_last=True):
        """
        Print the tree structure with proper lines and indentation.
        """
        # Prefix for current node
        print(prefix, "`-- " if is_last else "|-- ", self.data, sep="")

        # Determine the prefix for children
        new_prefix = prefix + ("    " if is_last else "|   ")

        # Recursively print each child
        for i, child in enumerate(self.child):
            is_child_last = (i == len(self.child) - 1)
            child.print_tree(new_prefix, is_child_last)



def graph():
    root = Treenode("computer science")

    #  Theoritical Foundation 
    foundation = Treenode("Theoritical Foundation")
    dsa = Treenode("data structure and algorithm")
    dms = Treenode("discrete mathematics")
    toc  = Treenode("automata theory")
    gt = Treenode("graph throey")

    data_structure = Treenode("data structure")
    data_structure.add_child_node(Treenode("arrays"))
    data_structure.add_child_node(Treenode("strings"))
    data_structure.add_child_node(Treenode("linked list"))
    data_structure.add_child_node(Treenode("queue"))
    data_structure.add_child_node(Treenode("hash maps"))
    data_structure.add_child_node(Treenode("trees"))
    data_structure.add_child_node(Treenode("graphs"))
    data_structure.add_child_node(Treenode("heaps"))


    data_algorithms = Treenode("data algorithms")
    data_algorithms.add_child_node(Treenode("sorting algorithms"))
    data_algorithms.add_child_node(Treenode("searching algorithms"))
    data_algorithms.add_child_node(Treenode("divide and conquer"))
    data_algorithms.add_child_node(Treenode("greedy"))
    data_algorithms.add_child_node(Treenode("dynamic programming"))
    data_algorithms.add_child_node(Treenode("backtracking"))
    data_algorithms.add_child_node(Treenode("graph algorithms"))

    dsa.add_child_node(data_structure)
    dsa.add_child_node(data_algorithms)

    dms.add_child_node(Treenode("set theory"))
    dms.add_child_node(Treenode("propositional and predicate logic"))
    dms.add_child_node(Treenode("combinatorics"))
    dms.add_child_node(Treenode("graph theory"))
    dms.add_child_node(Treenode("number theory"))
    dms.add_child_node(Treenode("boolean algebra"))
    dms.add_child_node(Treenode("probability"))
    dms.add_child_node(Treenode("algebraic structure"))
    dms.add_child_node(Treenode("automata and formal language"))
    dms.add_child_node(Treenode("application in computer science"))

    toc.add_child_node(Treenode("basics of automata theory"))
    toc.add_child_node(Treenode("finite automata"))
    toc.add_child_node(Treenode("regular language and expressions"))
    toc.add_child_node(Treenode("context free grammers"))
    toc.add_child_node(Treenode("pushdown automata"))
    toc.add_child_node(Treenode("turing machine"))
    toc.add_child_node(Treenode("chomsky hierarchy"))
    toc.add_child_node(Treenode("advanced topics"))


    gt.add_child_node(Treenode("basics of grpahs"))
    gt.add_child_node(Treenode("grpah representation"))
    gt.add_child_node(Treenode("graph properties"))
    gt.add_child_node(Treenode("graph traversal"))
    gt.add_child_node(Treenode("graph algorithms"))
    gt.add_child_node(Treenode("advanced graph concept"))
    gt.add_child_node(Treenode("application of graph theory"))
    gt.add_child_node(Treenode("graph theory in algorithms and problem solving"))  

    foundation.add_child_node(dsa)
    foundation.add_child_node(dms)
    foundation.add_child_node(toc)
    foundation.add_child_node(gt)



    #Software Development :)
    software_dev = Treenode("software development")
    prg_lang = Treenode("programming language")
    principle = Treenode("software engineering principles")
    prg_prdg = Treenode("programming paradigms")
    t_d = Treenode("testing and debugging")

    prg_lang.add_child_node(Treenode("python"))
    prg_lang.add_child_node(Treenode("java"))
    prg_lang.add_child_node(Treenode("c"))
    prg_lang.add_child_node(Treenode("c++"))
    prg_lang.add_child_node(Treenode("javascript"))

    software_dev.add_child_node(prg_lang)
    software_dev.add_child_node(principle)
    software_dev.add_child_node(prg_prdg)
    software_dev.add_child_node(t_d)




    system = Treenode("system")



    ai = Treenode("artificial intelligence")

    ml = Treenode("machine learning")
    neural_net = Treenode("neural netwoks")
    nlp = Treenode("natural language processing")
    cv = Treenode("computer vision")
    rl = Treenode("reinforcement learning")
    dl = Treenode("deep learning")
    ethics = Treenode("AI ethics and bias")
    robotics = Treenode("robotics")

    #machine learning

    f_ml = Treenode("foundational knowledge")
    p_l = Treenode("programming language for ml")
    mat_ml = Treenode("mathematics for ml")

    p_l.add_child_node(Treenode("pyhton"))
    p_l.add_child_node(Treenode("R"))

    mat_ml.add_child_node(Treenode("linear algebra"))
    mat_ml.add_child_node(Treenode("calculus"))
    mat_ml.add_child_node(Treenode("probability and statistics"))

    f_ml.add_child_node(p_l)
    f_ml.add_child_node(mat_ml)

    core_ml = Treenode("core machine leaning concepts")
    sl = Treenode("supervised learning")
    ul = Treenode("unsupervised learning")
    meo = Treenode("model evaluation and optimization")

    sl.add_child_node(Treenode("Regressions"))
    sl.add_child_node(Treenode("classification"))

    ul.add_child_node(Treenode("clustering"))
    ul.add_child_node(Treenode("dimensionality reduction"))

    meo.add_child_node(Treenode("metrics"))
    meo.add_child_node(Treenode("hyperparameter"))

    core_ml.add_child_node(sl)
    core_ml.add_child_node(ul)

    a_ml = Treenode("advanced topics")
    dl_ml = Treenode("deep leanring")
    nlp_ml = Treenode("natural language processing")
    r_ml = Treenode("reinforcement learning")

    dl_ml.add_child_node(Treenode("beural network"))
    dl_ml.add_child_node(Treenode("convolutional neural network"))
    dl_ml.add_child_node(Treenode("recurren neural network"))

    nlp_ml.add_child_node(Treenode("text processing"))
    nlp_ml.add_child_node(Treenode("sentiment analysis"))
    nlp_ml.add_child_node(Treenode("text generation"))

    r_ml.add_child_node(Treenode("agent and environmnet"))
    r_ml.add_child_node(Treenode("q learning"))

    a_ml.add_child_node(dl_ml)
    a_ml.add_child_node(r_ml)

    ml.add_child_node(f_ml)
    ml.add_child_node(core_ml)
    ml.add_child_node(a_ml)

    #neural network

    f_nn = Treenode("fundaments of neural networks")

    f_nn.add_child_node(Treenode("percetons"))
    f_nn.add_child_node(Treenode("multilayer perceptons"))
    f_nn.add_child_node(Treenode("activation function"))
    

    na_nn = Treenode("network achitecture")

    na_nn.add_child_node(Treenode("feedforward neural network"))
    na_nn.add_child_node(Treenode("convolutional neural network"))
    na_nn.add_child_node(Treenode("recurrent neural network"))
    na_nn.add_child_node(Treenode("advanced architecture"))
    na_nn.add_child_node(Treenode("generative adversarial networks"))

    t_nn = Treenode("trianing neural network")

    t_nn.add_child_node(Treenode("forward propogation"))
    t_nn.add_child_node(Treenode("backward propagation"))
    t_nn.add_child_node(Treenode("optimization algorithm"))
    t_nn.add_child_node(Treenode("loss function"))

    nr_nn = Treenode("network regularization")

    nr_nn.add_child_node(Treenode("overfitting and underfitting"))
    nr_nn.add_child_node(Treenode("regularization techniques"))

    hp_nn = Treenode("hyperparameter tuning")

    hp_nn.add_child_node(Treenode("learning rate"))
    hp_nn.add_child_node(Treenode("batch size"))
    hp_nn.add_child_node(Treenode("number of layers abd neurons"))
    hp_nn.add_child_node(Treenode("epochs"))


    tl_nn = Treenode("transfer learning")
    at_nn = Treenode("advanced topics")
    at_nn.add_child_node(Treenode("attention mechanism"))
    at_nn.add_child_node(Treenode("transfer models"))
    at_nn.add_child_node(Treenode("capsule networks"))


    neural_net.add_child_node(f_nn)
    neural_net.add_child_node(na_nn)
    neural_net.add_child_node(t_nn)
    neural_net.add_child_node(nr_nn)
    neural_net.add_child_node(hp_nn)
    neural_net.add_child_node(tl_nn)
    neural_net.add_child_node(at_nn)


    #natural language processing
    f_nlp = Treenode("fundaments of nlp")
    f_nlp.add_child_node(Treenode("basics of text processing"))
    f_nlp.add_child_node(Treenode("text summarization"))
    f_nlp.add_child_node(Treenode("understanding the corpora and dataset"))
    
    ssa_nlp = Treenode("syntatic and semantic analysis")
    ssa_nlp.add_child_node(Treenode("part of speech tagging"))
    ssa_nlp.add_child_node(Treenode("dependency parsing"))
    ssa_nlp.add_child_node(Treenode("named entity recognition"))
    ssa_nlp.add_child_node(Treenode("word sense disambiguation"))

    tswr_nlp = Treenode("text summarization and word representation")
    tswr_nlp.add_child_node(Treenode("text similarity"))
    tswr_nlp.add_child_node(Treenode("word_embaddings"))

    sm_nlp = Treenode("sequence modeling")
    sm_nlp.add_child_node(Treenode("language models"))
    sm_nlp.add_child_node(Treenode("recurrent neural networks"))
    sm_nlp.add_child_node(Treenode("lstm and gru's"))

    tam_nlp = Treenode("transformers and attention mechanism")
    tam_nlp.add_child_node(Treenode("attention mechanism"))
    tam_nlp.add_child_node(Treenode("transformers"))
    tam_nlp.add_child_node(Treenode("pretrained models"))
    tam_nlp.add_child_node(Treenode("fine tuning pretrained models"))

    a_nlp = Treenode("advances natural language processing")
    a_nlp.add_child_node(Treenode("sentiment analysis"))
    a_nlp.add_child_node(Treenode("text summarization"))
    a_nlp.add_child_node(Treenode("machine translation"))
    a_nlp.add_child_node(Treenode("question answering"))
    a_nlp.add_child_node(Treenode("text generation"))

    nlp.add_child_node(f_nlp)
    nlp.add_child_node(ssa_nlp)
    nlp.add_child_node(tswr_nlp)
    nlp.add_child_node(sm_nlp)
    nlp.add_child_node(tam_nlp)
    nlp.add_child_node(a_nlp)

    #computer vision
    f_cv = Treenode("fundamentals in computer vision")
    f_cv.add_child_node(Treenode("introduction to computer vision"))
    f_cv.add_child_node(Treenode("image basics"))
    f_cv.add_child_node(Treenode("mathematics for computeer vision"))
    f_cv.add_child_node(Treenode("image file formats"))

    ip_cv = Treenode("image processing")
    ip_cv.add_child_node(Treenode("basic image operations"))
    ip_cv.add_child_node(Treenode("filters"))
    ip_cv.add_child_node(Treenode("histogram analysis"))
    ip_cv.add_child_node(Treenode("color spaces"))

    gt_cv = Treenode("geometric transformations")
    gt_cv.add_child_node(Treenode("translation,roatiton and scaling"))
    gt_cv.add_child_node(Treenode("warping and morphing"))
    
    fdm_cv = Treenode("feature detection and matching")
    fdm_cv.add_child_node(Treenode("keypoint detection"))
    fdm_cv.add_child_node(Treenode("feature descriptors"))
    fdm_cv.add_child_node(Treenode("template matching"))

    dl_cv = Treenode("deep learning in computer vision")
    dl_cv.add_child_node(Treenode("convolutional neural networks"))
    dl_cv.add_child_node(Treenode("transfer learning"))
    dl_cv.add_child_node(Treenode("data augmentation"))
    dl_cv.add_child_node(Treenode("object detection"))
    dl_cv.add_child_node(Treenode("sematic and instance segmentation"))


    cv.add_child_node(f_cv)
    cv.add_child_node(ip_cv)
    cv.add_child_node(gt_cv)
    cv.add_child_node(fdm_cv)
    cv.add_child_node(dl_cv)
    

    #reinforcement learning
    rl.add_child_node(Treenode("fundamentals of reinforcement learning"))
    rl.add_child_node(Treenode("markov decision process"))
    rl.add_child_node(Treenode("value functions"))
    rl.add_child_node(Treenode("policies"))
    rl.add_child_node(Treenode("reinforcement learning algorithms"))
    rl.add_child_node(Treenode("deep reinforcement learning"))
    rl.add_child_node(Treenode("multi agent reinforcement learning"))
    rl.add_child_node(Treenode("evaluation and metrics"))

    #deep learning
    f_dl  = Treenode("foundation of deep learning")
    f_dl.add_child_node(Treenode("what is deep learning"))
    f_dl.add_child_node(Treenode("mathematics foundations"))
    f_dl.add_child_node(Treenode("artificial neural networks"))

    nnb_dl = Treenode("neural network basics")
    nnb_dl.add_child_node(Treenode("activation function"))
    nnb_dl.add_child_node(Treenode("forward and backpropogation"))
    nnb_dl.add_child_node(Treenode("loss function"))
    nnb_dl.add_child_node(Treenode("gradient descent"))
    nnb_dl.add_child_node(Treenode("weight intitalization"))

    tdnn_dl = Treenode("training deep learning neural networks")
    tdnn_dl.add_child_node(Treenode("optimization technique"))
    tdnn_dl.add_child_node(Treenode("regularization"))
    tdnn_dl.add_child_node(Treenode("hyperparameter tuning"))
    tdnn_dl.add_child_node(Treenode("vanishing and exploding gradients"))

    an_dl = Treenode("architectures and networks")
    an_dl.add_child_node(Treenode("feedforward neural network"))
    an_dl.add_child_node(Treenode("convolutional neural networks"))
    an_dl.add_child_node(Treenode("recurrent neural networks"))
    an_dl.add_child_node(Treenode("advanced architecture"))
    an_dl.add_child_node(Treenode("autoencodes"))
    an_dl.add_child_node(Treenode("generative adversarial networks"))

    dlf_dl = Treenode("deep learning frameworks")
    dlf_dl.add_child_node(Treenode("tensorflow"))
    dlf_dl.add_child_node(Treenode("pytorch"))
    dlf_dl.add_child_node(Treenode("jax"))

    dhp_dl = Treenode("data handling and preprocessing")
    dhp_dl.add_child_node(Treenode("dataset preparation"))
    dhp_dl.add_child_node(Treenode("data augmentation"))
    dhp_dl.add_child_node(Treenode("feature engineering"))

    s_dl = Treenode("specilaized deep learning topics")
    s_dl.add_child_node(Treenode("transfer learning"))
    s_dl.add_child_node(Treenode("reinforcement learning"))
    s_dl.add_child_node(Treenode("natural language processing"))
    s_dl.add_child_node(Treenode("computer visison"))

    dl.add_child_node(f_dl)
    dl.add_child_node(nnb_dl)
    dl.add_child_node(tdnn_dl)
    dl.add_child_node(an_dl)
    dl.add_child_node(dlf_dl)
    dl.add_child_node(dhp_dl)
    dl.add_child_node(s_dl)


    ai.add_child_node(ml)
    ai.add_child_node(neural_net)
    ai.add_child_node(nlp)
    ai.add_child_node(cv)
    ai.add_child_node(rl)
    ai.add_child_node(dl)
    ai.add_child_node(ethics)
    ai.add_child_node(robotics)

    dbms = Treenode("Data management")


    hci = Treenode("human computer interaction")

    root.add_child_node(foundation)
    root.add_child_node(software_dev)
    root.add_child_node(ai)
    root.add_child_node(dbms)
    root.add_child_node(hci)



    return root


if __name__ == "__main__":
    root = graph()  # Build the tree
    user_input = input("Enter the node value to display its tree structure: ").strip().lower()
    node = root.find_node(user_input)
    if node:
        print(f"Tree structure for '{user_input}':")
        node.print_tree()
    else:
        print(f"Node '{user_input}' not found in the graph.")

