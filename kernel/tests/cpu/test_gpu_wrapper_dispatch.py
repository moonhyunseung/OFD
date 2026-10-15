import unittest
from unittest.mock import patch
import torch
from ofd_kernel.contracts import Variant
from ofd_kernel.runtime.gpu_ofd import gpu_ofd_decode

class DispatchTest(unittest.TestCase):
    def test_all_variants_read_initialized_query(self):
        q=torch.tensor([1.0]);qw=torch.tensor([-999.0]);ow=torch.empty_like(q)
        for v in Variant:
            qw.fill_(-999)
            def transform(src,dst): dst.copy_(src*2)
            with patch('ofd_kernel.triton.fwht.fwht_into',side_effect=transform), patch('ofd_kernel.runtime.gpu_ofd.triton_paged_decode_core',side_effect=lambda query,*a,**kw:query.clone()) as core:
                out=gpu_ofd_decode(q,None,None,None,None,None,None,variant=v,config=None,workspace=None,max_seq_len=1,q_work=qw,out_work=ow)
                self.assertIs(core.call_args.args[0],qw if v.rotate_k else q)
                self.assertEqual(out.item(),(2 if v.rotate_k else 1)*(2 if v.rotate_v else 1))
