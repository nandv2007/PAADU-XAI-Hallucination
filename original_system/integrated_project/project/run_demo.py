import argparse,json
from integrated_pipeline import IntegratedSystem
p=argparse.ArgumentParser();p.add_argument('--nli',action='store_true',help='download and execute real NLI model');args=p.parse_args()
docs=[{'title':'UDP','text':'UDP is connectionless and does not guarantee reliable or ordered delivery.'},
      {'title':'TCP','text':'TCP is connection-oriented and provides reliable ordered data delivery.'},
      {'title':'DNS','text':'DNS maps domain names to IP addresses.'},
      {'title':'ARP','text':'ARP maps IPv4 addresses to MAC addresses on local networks.'}]
system=IntegratedSystem(docs,enable_nli=args.nli)
result=system.answer('Does UDP guarantee reliable ordered delivery?',
    'UDP is connectionless and does not guarantee reliable or ordered delivery.')
print(json.dumps(result,indent=2))
assert result['retrieval'] and result['retrieval'][0]['title']=='UDP'
if args.nli: assert result['verification']['label']=='SUPPORTED',result['verification']
print('PASS: reconstructed GraphRAG retrieval + integration interface'+(' + NLI' if args.nli else ''))
