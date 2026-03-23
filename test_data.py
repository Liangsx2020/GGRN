# test_data.py
from data import MMSDataGenerator, StencilCoefficientComputer

if __name__ == "__main__":
    print("Testing Base + MMS Generator...")
    gen = MMSDataGenerator(resolution=32)
    data = gen.build_graph(verbose=True)
    
    print("\nTesting Stencil Computation...")
    computer = StencilCoefficientComputer(max_order=2)
    data = computer.compute_stencils(data)
    
    print("\n✅ Success! Data object contains:")
    print(f"  x shape: {data.x.shape}")
    print(f"  edge_index shape: {data.edge_index.shape}")
    print(f"  coeff_lap shape: {data.coeff_lap.shape}")