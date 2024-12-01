import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from scipy.stats import *
from tqdm import tqdm



N_SIMULATIONS = 10000
N = 103

# Display the original data
def display_og_dist():
    types = ['Plain', 'Chocolate', 'Blueberry']
    og_data = {
        "Pancake type": types,
        "Frequencies": [24,19,60],
        "Proportions": [24/103,19/103,60/103]
    } 

    df = pd.DataFrame(og_data)


    chart = alt.Chart(df, title="Pancake orders by type").mark_bar().encode(
        x=alt.X("Pancake type"),
        y="Frequencies",
        color=alt.Color(
            "Pancake type",
            scale = alt.Scale(domain=types, range=['#F5F5DC', '#7B3F00', '#464196']),
            legend=None
        )
    )

    prop_chart = alt.Chart(df, title="Proportions of pancake orders by type").mark_bar().encode(
        x=alt.X("Pancake type"),
        y="Proportions",
        color=alt.Color(
            "Pancake type",
            scale = alt.Scale(domain=types, range=['#F5F5DC', '#7B3F00', '#464196']),
            legend=None
        )
    )
        
    st.altair_chart(chart, use_container_width=True)
    st.altair_chart(prop_chart, use_container_width=True)

    return {'plain': 24, 'chocolate': 19, 'blueberry': 60}

# Display new distribution with error bars
def display_new_dist(errs):
    types = ['Plain', 'Chocolate', 'Blueberry']
    og_data = {
        "Pancake type": types,
        "Frequency": [24,19,60],
        "Proportion": [24/103,19/103,60/103]
    } 

    df = pd.DataFrame(og_data)


    bar_chart = alt.Chart(df, title="Proportions of pancake orders by type").mark_bar().encode(
        x=alt.X("Pancake type"),
        y=alt.Y("Proportion"),
        color=alt.Color(
            "Pancake type",
            scale = alt.Scale(domain=types, range=['#F5F5DC', '#7B3F00', '#464196']),
            legend=None
        )
    )

    df['Error'] = df['Pancake type'].map(lambda x: errs[x])
    df['CI_Lower'] = -1 * df['Error']
    df['CI_Upper'] = df['Error']
    print(df)
    
    error_bars = alt.Chart(df).mark_errorbar().encode(
        x=alt.X('Pancake type:N'),
        y=alt.Y('Proportion:Q'),
        yError='CI_Upper:Q',  
        yError2='CI_Lower:Q',
        color = alt.value('red')
    )

    chart = bar_chart + error_bars
    st.altair_chart(chart, use_container_width=True)

    for pancake in types:
        error = round(float(df.loc[df['Pancake type'] == pancake, 'Error'].iloc[0]), 4)
        st.write(pancake + ' pancake error: ' + str(error))

# Perform bootstrapping
def run_bootstrap():
    results = {
        'blueberry': [],
        'chocolate': [],
        'plain': []
    }

    sample_set = ['plain'] * 24 + ['chocolate'] * 19 + ['blueberry'] * 60
    sample_size = len(sample_set)

    for i in tqdm(range(N_SIMULATIONS)):
        samples = np.random.choice(sample_set, size=sample_size, replace=True)
        for pancake in ['blueberry', 'chocolate', 'plain']:
            results[pancake].append(np.count_nonzero(samples == pancake) / sample_size)

    return pd.DataFrame(results)

# Plot bootstrapping results
def plot_bootstrapping(df):
    colors = {'blueberry': '#464196', 'chocolate': '#7B3F00', 'plain':  'black'}
    for pancake, color in colors.items():
        proportions, frequencies = np.unique(df[pancake], return_counts=True)
        plt.plot(proportions, frequencies, label=pancake.capitalize(), color=color, marker='o')

    plt.title("Bootstrapping Results", fontsize=16)
    plt.xlabel("Proportions", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(title="Pancake Type", fontsize=12)
    plt.grid(alpha=0.3)
    
    return plt

# Plot beta function
def plot_beta(a, b, title, color='black'):
    x = np.linspace(0, 1, 1000)
    y = beta.pdf(x, a, b)
    

    maximum = max(y)
    x_maximum = round(list(y).index(maximum) / 1000, 2)

    plt.clf()
    plt.plot(x, y, color=color)
    plt.plot(x_maximum, maximum, 'ro', label=f'Max: ({x_maximum}, {round(maximum, 2)})')
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Probability Density")
    plt.legend()
    st.pyplot(plt)

def main():
    st.title("Probability with Pancakes 🥞", anchor=False)
    st.write("Which pancake is the favorite among diners at the Palo Alto Opportunity Center, and by how much? Let's find out, using 109!")
    st.markdown("*For the best viewing results, please go to the top right corner and select dark mode under settings.*")
    left, mid, right = st.columns(3)
    with mid:
        st.image('Breakfast-flip.gif')
    st.divider()
    st.write("Here's the raw data I obtained from one serving period of the types of pancakes ordered: ")

    og_dataset = display_og_dist()

    st.write("This might be a good measurement of which pancakes diners actually like, but where's the uncertainty? How about diners who weren't part of my initial sample? Let's use bootstrapping to find the uncertainty in our data.")

    if st.button('Bootstrap!', key="bootstrap_button"):
        st.write(f'Bootstrapping with {N_SIMULATIONS:,} trials of sample size 103...')
        with st.spinner(''):
            data = run_bootstrap()
            plot = plot_bootstrapping(data)
            st.pyplot(plot)
        
            st.divider()
            st.write("Now, let's incorporate this information to find the standard error in our data!")
            blueberries = list(map(lambda x: x, data['blueberry'].tolist()))
            plains = list(map(lambda x: x, data['plain'].tolist()))
            chocos = list(map(lambda x: x, data['chocolate'].tolist()))
            cis = {
                "Blueberry": np.std(blueberries, ddof=1) / np.sqrt(N),
                "Plain": np.std(plains, ddof=1) / np.sqrt(N), 
                "Chocolate": np.std(chocos, ddof=1) / np.sqrt(N)
            }
            print(cis)
            display_new_dist(cis)
            st.write("Alright, so we're pretty confident that we should make pancakes at a 58:19:23 ratio! \
                     But how could we explore this information further, using 109 concepts?")

    st.divider()

            
    st.write("From just the concepts learned in class, we could model this situation as a beta distribution: \
                for each pancake type, a person has a certain probability of ordering it. \
                When focusing on blueberry pancakes, an order of a blueberry pancake, for example, would be considered a success, \
                while any other pancake is a 'failure.'")
    
    st.write("Anecdotally, we believe in a 6:2:1 ratio of blueberry:plain:chocolate, which we could incorporate into our beta distributions under a Bayesian approach. With the frequentist approach, we begin with a uniform distribution.")
    
    st.write("Check out the beta distributions of the different types of pancakes!")

    beta_select = st.segmented_control("Beta Approach", ["Bayesian", "Frequentist"], selection_mode="single", default="Bayesian")
    initial_blue, initial_plain, initial_chocolate, initial_blue_f, initial_plain_f, initial_chocolate_f = 1,1,1,1,1,1
    if beta_select == "Bayesian":
        strength = st.slider("Strength of belief in prior distribution", 1, 100, 1)
        initial_blue = 6 * strength
        initial_plain = 2 * strength
        initial_chocolate = 1 * strength
        initial_blue_f = initial_plain + initial_chocolate
        initial_plain_f = initial_blue + initial_chocolate
        initial_chocolate_f = initial_blue + initial_plain

    plot_beta(60 + initial_blue, 43 + initial_blue_f, "Blueberry Pancakes", color='#464196')
    plot_beta(24 + initial_plain, 81 + initial_plain_f, "Plain Pancakes")
    plot_beta(19 + initial_chocolate, 84 + initial_chocolate_f, "Chocolate Pancakes", color='#7B3F00')

    st.write("The Dirichlet distribution can unify these probabilities into one: essentially, it is a beta distirbution \
                for multinomials. Rather than treating these orders as binomials, we can treat them as multinomials, with \
                certain probabilities of ordering either a blueberry, plain, or chocolate pancake.")
    
    dirichlet_mode = st.segmented_control("Dirichlet Approach", ["Bayesian", "Frequentist"], selection_mode="single", default="Bayesian", key="dirichlet_mode")
    l, c, r = st.columns([1,4,1])
    imagefile = "dirichlet_dist.png" if dirichlet_mode == "Frequentist" else "dirichlet_dist_bayesian.png"
    with c:
        with st.spinner(''):
            st.image(imagefile)
    st.write('"Hotter" colors represent higher probability densities, and the probabilities increase from left to right on each \
                side of the triangle. From the left-most side counter-clockwise, the sides are blueberry, plain, and chocolate pancakes. ')
    st.markdown('*Image created from code provided by Thomas Boggs (tboggs on GitHub)')
            
    st.divider()

    st.subheader('Preferred Pancake Predictor', anchor=False)
    
    predictor_mode = st.segmented_control("Which approach would you like to use?", ["Bayesian", "Frequentist"], selection_mode="single", default="Bayesian", key="predictor_mode")
    inits = {"b": 1, "p": 1, "c": 1}
    if predictor_mode == "Bayesian":
        predictor_strength = st.slider("Strength of belief in prior distribution", 1, 100, 1, key="predictor_strength")
        inits = {"b": 6 * predictor_strength, "p": 2 * predictor_strength, "c": 1 * predictor_strength}

    n_pancakes = st.number_input(
        "How many total pancakes can you make?", value=None, placeholder="Type a number..."
    )
    if n_pancakes:
        alpha = [60 + inits["b"], 24 + inits["p"], 19 + inits["c"]]

        sample_p = dirichlet.rvs(alpha, size=1)[0]

        blue_p = sample_p[0]
        plain_p = sample_p[1]
        choc_p = sample_p[2]

        ps = [blue_p, plain_p, choc_p]
        sample = multinomial.rvs(n_pancakes, ps)
        blueberrry_part = "blueberry pancake," if sample[0] == 1 else "blueberry pancakes,"
        plain_part = "plain pancake, and" if sample[1] == 1 else "plain pancakes, and"
        chocolate_part = "chocolate pancake!" if sample[2] == 1 else "chocolate pancakes!"
        st.write("You should make", sample[0], blueberrry_part, sample[1], plain_part, sample[2], chocolate_part)

    st.divider()

    st.subheader("Sources and More Information")
    st.write("This project was created through lessons learned from CS109!")
    url = 'https://www.sciencedirect.com/topics/mathematics/dirichlet-distribution'
    st.write("The Dirichlet visualization was made possible through open-source code from Thomas Boggs (tboggs on GitHub).\
                You can learn more about the Dirichlet distribution [here](%s)." % url)
    opportunity_url = 'https://www.lifemoves.org/directory/opportunity-services-center/'
    st.write("The Breakfast Club partners with the LifeMoves Palo Alto Opportunity Center, so please visit their [website](%s) for more information!" % opportunity_url)
    st.write("You can read more about this project in my project write-up!")







if __name__ == '__main__':
    main()