use core::iter::Iterator;

use p3_field::FieldAlgebra;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use zkm_core_executor::{ExecutionRecord, Opcode, events::AluEvent};
use zkm_core_machine::alu::{AddSubChip, NUM_ADD_SUB_COLS};
use zkm_core_machine::utils::pad_rows_fixed;
use zkm_stark::air::MachineAir;

use kernels::*;

pub fn test_generate_trace_cuda_eq_cpu() {
    let _context = init_cuda();

    let shard = {
        let add_events = (0..1)
            .flat_map(|i| {
                [{
                    let operand_1 = 1u32;
                    let operand_2 = 2u32;
                    let result = operand_1.wrapping_add(operand_2);
                    AluEvent::new(i % 2, Opcode::ADD, result, operand_1, operand_2)
                }]
            })
            .collect::<Vec<_>>();

        ExecutionRecord {
            add_events,
            ..Default::default()
        }
    };

    let chip = AddSubChip;
    let trace: RowMajorMatrix<KoalaBear> =
        chip.generate_trace(&shard, &mut ExecutionRecord::default());
    let trace_cuda = generate_trace_cuda(shard);

    assert_eq!(trace_cuda, trace);
    println!("trace_cuda = trace");
}

fn generate_trace_cuda(input: ExecutionRecord) -> RowMajorMatrix<KoalaBear> {
    type F = KoalaBear;

    let events = input
        .add_events
        .into_iter()
        .chain(input.sub_events)
        .collect::<Vec<_>>();

    let d_events = CudaBuffer::<AluEvent>::copy_from("add_sub_events", &events);
    let d_rows = CudaBuffer::<[F; NUM_ADD_SUB_COLS]>::new("AddSubCols", events.len());

    ffi_wrap(|| unsafe {
        add_sub_events_to_rows(
            d_events.as_device_ptr(),
            d_rows.as_device_ptr(),
            events.len(),
        )
    })
    .unwrap();

    let mut rows = d_rows.to_vec();

    pad_rows_fixed(&mut rows, || [F::ZERO; NUM_ADD_SUB_COLS], None);

    // Convert the trace to a row major matrix.
    RowMajorMatrix::new(
        rows.into_iter().flatten().collect::<Vec<_>>(),
        NUM_ADD_SUB_COLS,
    )
}
