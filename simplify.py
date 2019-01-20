#!/usr/bin/python3
 # -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import unidecode
import string
import re
import torch.nn.functional as F


#TODO BATCHES
#TODO POOLING
#TODO HIGHWAY

options = {'dec_hidden':100, 'enc_hidden':100, 'emb_dim': 100, 'pooling':5, 'vocab_size':50, 'max_conv_kern':4}

class Encoder(nn.Module):
    def __init__(self, options):
        super(Encoder, self).__init__()
        self.hidden_size = options['enc_hidden']
        self.emb_size = options['emb_dim']
        self.batch_sizes = options['batch_sizes']
        self.filters = [(i,100) for i in range(1,options['max_conv_kern'],2)]
        self.embeddings = nn.Embedding(options['vocab_size'],options['emb_dim'])
        self.gru = nn.GRU(options['conv_out'], options['enc_hidden'], bias = True)
        self.convolutionallayers = []
        for w,j in self.filters:
            for i in range(j):
                self.convolutionallayers.append(nn.Conv1d(options['emb_dim'], 1, w, stride=1, dilation=1,
                                                          padding=(int(w/2)+1), groups=1, bias=True))
        self.convs = nn.Sequential(*self.convolutionallayers)

    def forward(self, hidden, inp):
        recurrent_inp = self.embeddings(inp)
        convolutions = []
        for conv_layer in self.convs:
                convolutions.append(conv_layer(recurrent_inp.view(self.batch_sizes,self.emb_size,-1)))
        recurrent_inp = torch.cat(convolutions)
        recurrent_inp = F.relu(recurrent_inp)
        h_n, h_t = self.gru(recurrent_inp.view(self.batch_sizes,1,600),
                            (hidden.view(self.batch_sizes,1, self.hidden_size)))
        return h_n, h_t

    def reset_hidden(self):
        hidden = Variable(torch.zeros(self.batch_sizes, 1, self.hidden_size))
        return hidden


class Decoder(nn.Module):
    def __init__(self, options,emb_size, batch_sizes = 1):
        super(Decoder, self).__init__()
        self.hidden_size = options['dec_hidden']
        self.emb_size = 200
        self.batch_sizes = batch_sizes
        self.embeddings = nn.Embedding(options['vocab_size'],200)
        self.w_out = nn.Linear(options['dec_hidden'], options['dec_out'])
        self.GRU = nn.GRU(self.emb_size, options['enc_hidden'], bias = True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp, hidden):
        inp = self.embeddings(inp)
        output, hidden = self.GRU(inp.view(-1, 1, self.emb_size), (hidden.view(self.batch_sizes, 1, -1)))
        output = self.softmax(self.w_out(output[0]))
        return output, hidden

    def reset_hidden(self):
        hidden = Variable(torch.zeros(self.batch_sizes, 1, self.hidden_size))
        return hidden

class Encoder_Decoder():
    def __init__(self, options, batch_sizes =1):
        self.batch_sizes = batch_sizes
        self.n_characters = len(string.printable)
        self.decoder = Decoder(options, batch_sizes)
        self.encoder = Encoder(options)
        self.eos_token = '#'
        self.sos_token = '$'
        self.dec_optimizer = optim.SGD(self.decoder.parameters(), lr = 0.02, momentum=0.01)
        self.enc_optimizer = optim.SGD(self.encoder.parameters(), lr = 0.02, momentum=0.01)

    def single_forward(self, inps):
        hidden = self.encoder.reset_hidden()
        for inp in inps:
            h_f, hidden = self.encoder(hidden,inp)
            h_b = h_f
            idx = [i for i in range(h_b.size(0)-1, -1, -1)]
            idx = Variable(torch.LongTensor(idx))
            h_b = h_b.index_select(0, idx)
            h = torch.cat([h_f,h_b], dim = -1)
        return h_f, hidden

    def single_backward(self, last_hidden, h_cont, target, teacher_forcing = False):
        loss = 0
        #self.decoder.hidden = self.encoder.hidden
        #self.decoder.last = self.char_tensor(self.eos_token)
        self.dec_optimizer.zero_grad()
        self.enc_optimizer.zero_grad()
        #self.att_optimizer.zero_grad()
        hidden = self.decoder.reset_hidden()
        hidden = torch.cat([last_hidden, last_hidden])
        criterion = nn.CrossEntropyLoss()
        inp = self.char_tensor(self.sos_token)
        if teacher_forcing == True:
            for idx in range(len(target)-1):
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                out, last_hidden = self.decoder(inp,last_hidden)
                current_target = target[idx + 1]
                loss += criterion(out, current_target.unsqueeze(0))
                _,top = out.data.topk(1)
                inp = target[idx+1]
            self.enc_optimizer.step()
            self.dec_optimizer.step()
            #self.att_optimizer.step()
            loss.backward()
        else:
            for idx in range(len(target)-1):
                self.decoder.zero_grad()
                self.encoder.zero_grad()
                #self.decoder.attention.zero_grad()
                out, last_hidden = self.decoder(inp,last_hidden)
                current_target = target[idx+1]
                loss += criterion(out, current_target.unsqueeze(0))
                _,top = out.data.topk(1)
                inp = Variable(top)
            loss.backward()
            self.enc_optimizer.step()
            self.dec_optimizer.step()
            #self.att_optimizer.step()

    def read_file(self, filename):
        file = unidecode.unidecode(open(filename).read())
        return file

    def char_tensor(self,sentence):
        all_characters = string.printable
        tensor = torch.zeros(len(sentence)).long()
        for c in range(len(sentence)):
            tensor[c] = all_characters.index(sentence[c])
        return Variable(tensor)

    def input_target(self, line):
        assert(len(line) == 2)
        assert(isinstance(line[0],str))
        assert(isinstance(line[1],str))
        fw_inp = self.char_tensor(line[0])
        bw_inp = self.char_tensor(line[0][::-1])
        target = self.char_tensor(line[1])
        return fw_inp, bw_inp, target

    def train(self, epochs, batches, line):
        for i in range(epochs):
            for input_target in line:
                i_t = re.split(r'\t', input_target)
                inp, _, trg = self.input_target(i_t)
                out, last_hidden = self.single_forward(inp)
                print('encoding ', i, ' done!')
                stats = self.single_backward(last_hidden, out, trg)
                print('decoding ', i, ' done!')
                predicted = self.generate()
                print(predicted)
        return stats

    def generate(self, primer = False):
        maxlen = 10
        if not primer:
            primer = self.sos_token
        h, last_hidden = self.single_forward(self.char_tensor(primer))
        dec_hidden = self.decoder.reset_hidden()
        hidden = torch.cat([last_hidden,last_hidden])
        predicted = ''
        #while inp != self.char_tensor(self.eos_token) and len(predicted)<40:
        #for i in range(4):
        inp = self.char_tensor(self.sos_token)
        while len(predicted) < maxlen:
            out, last_hidden = self.decoder(inp,last_hidden)
            _, top = out.data.topk(1)
            charidx = top[0][0]
            nextchar = string.printable[charidx]
            predicted += nextchar
            #print(predicted)
            inp = self.char_tensor(nextchar)
        return predicted

    def evaluate(self,generated,target,n):
        for m in range(n):
            for i in range(len(target)-n):
                target_grams = generated_grams.append((generated[i+m],generated[i+m+1]))
            for i in range(len(generated)-n):
                generated_grams.append((generated[i+m],generated[i+m+1]))


if __name__ == '__main__':
    dev = """$Go.#	$Va !#
$Run!#	$Cours !#
$Run!#	$Courez !#
$Wow!#	$Ca alors !#
$Fire!#	$Au feu !#
$Help!#	$A l'aide !#
$Jump.#	$Saute.#
$Stop!#	$Ca suffit !#
$Stop!#	$Stop !#
$Stop!#	$Arrete-toi !#
$Wait!#	$Attends !#
$Wait!#	$Attendez !#"""
    devv = """$Hi.	Salut !#
$Run!	Cours !#
$Run!	Courez !#
$Wow!	Ca alors !#
$Fire!	Au feu !#
$Help!	A l'aide !#
$Jump.	Saute.#
$Stop!	Ca suffit !#
$Stop!	Stop !#
$Stop!	Arrete-toi !#
$Wait!	Attends !#
$Wait!	Attendez !#
$Go on.	Poursuis.#
$Go on.	Continuez.#
$Go on.	Poursuivez.#
$Hello!	Bonjour !#
$Hello!	Salut !#
$I see.	Je comprends.#
$I try.	J'essaye.#
$I won!	J'ai gagne !#
$I won!	Je l'ai emporte !#
$Oh no!	Oh non !#
$Attack!	Attaque !#
$Attack!	Attaquez !#
$Cheers!	Sante !#
$Cheers!	A votre sante !#
$Cheers!	Merci !#
$Cheers!	Tchin-tchin !#
$Get up.	Leve-toi.#
$Go now.	Va, maintenant.#
$Go now.	Allez-y maintenant.#
$Go now.	Vas-y maintenant.#
$Got it!	J'ai pige !#
$Got it!	Compris !#
$Got it?	Pige ?#
$Got it?	Compris ?#
$Got it?	T'as capte ?#
$Hop in.	Monte.#
$Hop in.	Montez.#
$Hug me.	Serre-moi dans tes bras !#
$Hug me.	Serrez-moi dans vos bras !#
$I fell.	Je suis tombee.#
$I fell.	Je suis tombe.#
$I know.	Je sais.#
$I left.	Je suis parti.#
$I left.	Je suis partie.#
$I lost.	J'ai perdu.#
$I'm 19.	J'ai 19 ans.#
$I'm OK.	Je vais bien.#
$I'm OK.	Ca va.#
$Listen.	Ecoutez !#
$No way!	C'est pas possible !#
$No way!	Impossible !#
$No way!	En aucun cas.#
$No way!	Sans facons !#
$No way!	C'est hors de question !#
$No way!	Il n'en est pas question !#
$No way!	C'est exclu !#
$No way!	En aucune maniere !#
$No way!	Hors de question !#
$Really?	Vraiment ?#
$Really?	Vrai ?#
$Really?	Ah bon ?#
$Thanks.	Merci !#
$We try.	On essaye.#
$We won.	Nous avons gagne.#
$We won.	Nous gagnames.#
$We won.	Nous l'avons emporte.#
$We won.	Nous l'emportames.#
$Ask Tom.	Demande a Tom.#
$Awesome!	Fantastique !#
$Be calm.	Sois calme !#
$Be calm.	Soyez calme !#
$Be calm.	Soyez calmes !#
$Be cool.	Sois detendu !#
$Be fair.	Sois juste !#
$Be fair.	Soyez juste !#
$Be fair.	Soyez justes !#
$Be fair.	Sois equitable !#
$Be fair.	Soyez equitable !#
$Be fair.	Soyez equitables !#
$Be kind.	Sois gentil."""
    dev = dev.split('\n')
    options = {'dec_hidden': 100, 'enc_hidden': 100, 'emb_dim': 100, 'pooling': 5, 'vocab_size': 105,
               'max_conv_kern': 4, 'conv_out': 600, 'dec_out': 100, 'batch_sizes': 1}
    end_dec = Encoder_Decoder(options)
    end_dec.encoder.zero_grad()
    end_dec.decoder.zero_grad()
    end_dec.train(100,1,dev)
    predicted = end_dec.generate(primer = "$Help!#")
    print(predicted)
