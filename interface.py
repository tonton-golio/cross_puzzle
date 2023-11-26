import streamlit as st
import plotly.graph_objects as go
import numpy as np
import itertools

# Util functions

st.set_page_config(layout="wide")

@st.cache_resource()
class PeaceMaker:
    def __init__(self):
        self.base_piece = np.ones((8, 2, 2))

        self.to_zeroes = {
                            0 : [(2,1,0),(3,1,0),(2,1,1), (3,1,1)],
                            1 : [(2,1,0),(2,1,1),(3,1,1), (3,0,1), (4,0,1), (4,1,1)],
                            2 : [(2,1,0),(3,1,0),(2,1,1), (3,0,0), (4,0,0), (4,1,0), (3,1,1)],
                            3 : [(2,0,1),(2,1,1),(3,0,0), (3,0,1), (4,0,0), (4,0,1), (4,1,1), (5,0,1), (5,1,1)],
                            4 : [(3,0,0),(3,0,1),(3,1,1), (4,0,1), (4,0,0)],
                            5 : [(2,1,0),(2,1,1),(3,0,1), (3,1,0), (3,1,1), (4,0,1), (4,1,1), (5,1,1), (5,1,0)],
            }
        
        self.rotation_permutations = list(itertools.product(*([0, 90, 180, 270] for _ in range(3))))
        
        self.pieces = {}
        for k, v in self.to_zeroes.items():
            piece = self.make_piece(v)
            expanded_piece = self.piece_expander(piece)
            if k == 0:
                piece_permutations = self.get_piece_permutations(expanded_piece)[:4]
            else:
                piece_permutations = self.get_piece_permutations(expanded_piece)

            self.pieces[k] = {
                'piece': piece,
                'expanded_piece': expanded_piece,
                'piece_permutations': piece_permutations
            }

    def make_piece(self, z):
        piece = self.base_piece.copy()
        for i in z:
            piece[i] = 0
        return piece

    def piece_expander(self, piece):
        # take a piece which is like 8,2,2 and place in 8,8,8 grid 
        # should be centered in the middle of the grid

        expanded_piece = np.zeros((8,8,8))
        x,y,z = piece.shape
        expanded_piece[4-x//2:4+x//2,4-y//2:4+y//2,4-z//2:4+z//2] = piece

        return expanded_piece
      
    def rotate_piece(self, piece, axis, degrees=90):
        """
        Rotate piece around axis by d degrees (will be 0, 90, 180, 270)
        """
        axes = {'x': (1,2), 'y': (0,2), 'z': (0,1)}
        axis = axes[axis]
        k = degrees // 90
        return np.rot90(piece, k=k, axes=axis)

    def get_translations(self, expanded_piece):
        def get_long_dim(expanded_piece):
            # get the long dimension of the piece
            # long dimension is the dimension with no zeros
            dim_sum = []
            for ax in (0,1,2):
                other_axes = [0,1,2]
                other_axes.remove(ax)
                dim_sum.append(np.sum(expanded_piece, axis=tuple(other_axes)))

            # long_dim is the dimsum with no zeros
            long_dim = None
            for i, d in enumerate(dim_sum):
                if 0 not in d:
                    long_dim = i
                    break
            return long_dim

        long_dim = get_long_dim(expanded_piece)

        short_dims = [0,1,2]
        short_dims.remove(long_dim)

        # roll along short dims
        rolled = []

        rolled.append(np.roll(expanded_piece, shift=1, axis=short_dims[0]))
        rolled.append(np.roll(expanded_piece, shift=-1, axis=short_dims[0]))
        rolled.append(np.roll(expanded_piece, shift=1, axis=short_dims[1]))
        rolled.append(np.roll(expanded_piece, shift=-1, axis=short_dims[1]))
        return rolled

    def get_piece_permutations(self, piece):
        """
        Get all possible permutations of a piece
        """
        piece_origial = piece.copy()
        piece_permute = piece.copy()

        piece_permutations = []
        for p in self.rotation_permutations:
            piece_permute = piece_origial.copy()
        
            piece_permute = self.rotate_piece(piece_permute, axis='x', degrees=p[0])
            piece_permute = self.rotate_piece(piece_permute, axis='y', degrees=p[1])
            piece_permute = self.rotate_piece(piece_permute, axis='z', degrees=p[2])

            # get translations
            translations = self.get_translations(piece_permute)
            for t in translations:
                piece_permutations.append(t)
                    
        return piece_permutations

def animate_individual_pieces(pieces):
    # define 6 colors
    colors = ['red', 'green', 'blue', 'yellow', 'orange', 'purple']
    colors = ['salmon']*6

    for pn, c in zip(pieces.keys(), colors):
        piece = pieces[pn]['piece']

        import matplotlib.pyplot as plt
        from celluloid import Camera

        # make assets/piece1 folder if missing
        import os
        if not os.path.exists('assets/piece'+str(pn)):
            os.makedirs('assets/piece'+str(pn))

        

        Rand_start_angle = [np.random.randint(20,60),np.random.randint(0,180)]
        for i in range(360):
            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.voxels(piece, edgecolor='k', facecolor=c, alpha=0.96)
            ax.set_box_aspect((4,1,1))
            ax.set_axis_off()

            # Update view angle
            angle = Rand_start_angle[0] + i * 2  # Adjust the multiplier for speed/extent of rotation
            # print(angle)
            ax.view_init(Rand_start_angle[1], angle) # this doesnt work for some reason...
            
            # remove background
            ax.set_facecolor('none')

            plt.savefig(f'assets/piece{pn}/{i}.png', bbox_inches='tight')
            plt.close()
        # if file exists, delete it
        if os.path.exists('assets/animations/piece'+str(pn)+'.gif'):
            os.remove('assets/animations/piece'+str(pn)+'.gif')
            
        command = 'ffmpeg -r 30 -i assets/piece'+str(pn)+'/%d.png -vf "fps=25,format=rgb24,scale=328:328:flags=lanczos" -c:v gif assets/animations/piece'+str(pn)+'.gif'
        import subprocess
        subprocess.run(command, shell=True)

def is_valid_combination(ps):
    comb = np.zeros((8,8,8))
    for p in ps:
        comb += p
    if np.max(comb) < 2:
        return True
    else:
        return False
    
def get_valid_combinations(P1, P2):
    valid_combinations = []
    valid_combinations_indices = []
    for i, p1 in enumerate(P1):
        for j, p2 in enumerate(P2):
            if is_valid_combination((p1, p2)):
                valid_combinations_indices.append((i,j))
                valid_combinations.append(p1 + p2)

    return valid_combinations_indices, valid_combinations

@st.cache_data(ttl=60*60)
def do_2_piece_combinations(pieces):
    # combine pieces, adding one at a time
    st.write("## 2 Piece combinations")
    P1 = pieces[0]['piece_permutations']
    all_valid_combination_indices = []
    for i in range(5):          
        P2 = pieces[i+1]['piece_permutations']
        valid_combinations_indices, valid_combinations = get_valid_combinations(P1, P2)
        all_valid_combination_indices.append(valid_combinations_indices)

        st.write(f"**{i+2} Pieces** yielded; number of valid combinations: ", len(valid_combinations_indices), 'out of ', len(P1) * len(P2))

        P1 = valid_combinations

    return valid_combinations_indices, valid_combinations, all_valid_combination_indices

# Plotting functions
def plot_piece_grid(piece, x_offset=0, fig=None):

    # Define the vertices of a unit cube
    vertices = np.array([[0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1]])

    # Define the edges of the cube
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),
            (0, 4), (1, 5), (2, 6), (3, 7),
            (4, 5), (5, 6), (6, 7), (7, 4)]

    # Create the 3D plot
    if fig is None:
        fig = go.Figure()

    # Add cubes to the plot based on the shape
    # Add cubes to the plot based on the shape
    for x in range(piece.shape[0]):
        
        for y in range(piece.shape[1]):
            for z in range(piece.shape[2]):
                if piece[x, y, z] == 1:
                    # Calculate the offset for each cube
                    offset = np.array([x, y, z])
                    offset += np.array([x_offset, 0, 0])
                    # Add the edges to the plot for each cube with color
                    for edge in edges:
                        fig.add_trace(go.Scatter3d(x=vertices[list(edge), 0] + offset[0],
                                                y=vertices[list(edge), 1] + offset[1],
                                                z=vertices[list(edge), 2] + offset[2],
                                                mode='lines',
                                                line=dict(color='blue')))  # Specify the color here
                        
                if piece[x, y, z] > 1:
                    # Calculate the offset for each cube
                    offset = np.array([x, y, z])
                    # Add the edges to the plot for each cube with color
                    for edge in edges:
                        fig.add_trace(go.Scatter3d(x=vertices[list(edge), 0] + offset[0],
                                                y=vertices[list(edge), 1] + offset[1],
                                                z=vertices[list(edge), 2] + offset[2],
                                                mode='lines',
                                                line=dict(color='red')))

    # Set plot layout
    fig.update_layout(title='3D Composite Shape', 
                                                    # scene=dict(xaxis=dict(nticks=4, range=[-1, 9]),
                                                    #         yaxis=dict(nticks=4, range=[-1, 3]),
                                                    #         zaxis=dict(nticks=4, range=[-1, 3]))
                                                            )
    # remove legend
    fig.update_layout(showlegend=False)

    # set figure size
    fig.update_layout(
        autosize=False,
        width=1000,
        height=500,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
    return fig

def plot_multiple_pieces(pieces):
    fig = go.Figure()
    for k, v in pieces.items():
        fig = plot_piece_grid(v['piece'], x_offset=k*10, fig=fig)
    fig.update_layout(title='3D Composite Shape', 
                                                    scene=dict(xaxis=dict(nticks=4, range=[-1, 81]),
                                                             yaxis=dict(nticks=4, range=[-1, 81]),
                                                             zaxis=dict(nticks=4, range=[-1, 81]))
                                                            )
    # set figure size
    fig.update_layout(
        autosize=False,
        width=1300,
        height=800,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        )
    )
    return fig


if __name__ == "__main__":
    
    # header
    st.title("3D Composite Shape Explorer")
    
    # view gifs
    cols = st.columns(6)
    for i, c in enumerate(cols):
        with c:
            try:
                st.image('assets/animations/piece{}.gif'.format(i))
            except:
                pass

    # make pieces
    pieces = PeaceMaker().pieces

    cols = st.columns((1,3))
    with cols[0]:
        valid_combinations_indices, valid_combinations, all_valid_combination_indices = do_2_piece_combinations(pieces)

    # pick one
    last0 = all_valid_combination_indices[-1][0]
    last1 = all_valid_combination_indices[-2][last0[0]]
    last2 = all_valid_combination_indices[-3][last1[0]]
    last3 = all_valid_combination_indices[-4][last2[0]]
    last4 = all_valid_combination_indices[-5][last3[0]]
    final_indicies = [last4[0], last4[1], last3[1], last2[1], last1[1], last0[1]]

    # print a valid combination
    st.write("### Example valid combination")

    with cols[1]:
        fig = plot_piece_grid(valid_combinations[0])
        st.plotly_chart(fig)

    final_pieces = []
    for i, v in enumerate(final_indicies):
        # st.write(i,v)
        single_piece = pieces[i]['piece_permutations'][v]
        final_pieces.append(single_piece)

    final_pieces = {i+1: {'piece': v} for i, v in enumerate(final_pieces)}
    fig = plot_multiple_pieces(final_pieces)
    st.plotly_chart(fig)
