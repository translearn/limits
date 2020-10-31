from sympy.printing.ccode import C99CodePrinter
from sympy.utilities.codegen import C99CodeGen, ResultBase, CCodeGen, Result
from sympy.printing.codeprinter import AssignmentError
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.codegen.ast import real, complex128, float64

'''
Note: This file is based on code provided by SymPy
Copyright (c) 2006-2020 SymPy Development Team

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of SymPy nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
'''


class ComplexPrinter(C99CodePrinter):

    def __init__(self, settings=None):
        super().__init__(settings)
        self.known_functions['sqrt'] = 'csqrt'
        

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)
        if expr.exp == -1:
            return '1.0%s/%s' % (suffix.upper(), self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return '%scsqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%scpow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                         self._print(expr.exp))


class ComplexC99CodeGen(C99CodeGen):

    def __init__(self, project="project", printer=None,
                 preprocessor_statements=None, cse=False, print_result=False):

        CCodeGen.__init__(self, project=project, printer=printer,
                          preprocessor_statements=preprocessor_statements, cse=cse)

        self.preprocessor_statements.append('#include <complex.h>')
        
        
        self.print_result = print_result

        if self.print_result:
            self.preprocessor_statements.append('#include <stdio.h>')  

    def get_prototype(self, routine):
        
        if len(routine.results) > 1:
            raise ValueError("C only supports a single or no return value.")
        elif len(routine.results) == 1:
            
            ctype = 'double complex'
        else:
            ctype = "void"

        type_args = []
        for arg in routine.arguments:
            name = self.printer.doprint(arg.name)
            if arg.dimensions or isinstance(arg, ResultBase):
                type_args.append((arg.get_datatype('C'), "*%s" % name))
            else:
                type_args.append((arg.get_datatype('C'), name))
        arguments = ", ".join(["%s %s" % t for t in type_args])
        return "%s %s(%s)" % (ctype, routine.name, arguments)

    def _call_printer(self, routine):
        code_lines = []

        
        
        
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        return_val = None
        for result in routine.result_variables:
            if isinstance(result, Result):
                assign_to = routine.name + "_result"
                
                t = 'double complex'
                code_lines.append("{0} {1};\n".format(t, str(assign_to)))
                return_val = assign_to
            else:
                assign_to = result.result_var

            try:
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', dict(human=False, dereference=dereference),
                    result.expr, assign_to=assign_to)
            except AssignmentError:
                assign_to = result.result_var
                code_lines.append(
                    "%s %s;\n" % (result.get_datatype('c'), str(assign_to)))
                constants, not_c, c_expr = self._printer_method_with_settings(
                    'doprint', dict(human=False, dereference=dereference),
                    result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))
            code_lines.append("%s\n" % c_expr)

        
        if self.print_result:
            code_lines.append('printf("%f%+fi\\n", crealf(' + return_val + '), cimagf(' + return_val + '));\n')

        if return_val:
            code_lines.append("   return %s;\n" % return_val)
        return code_lines

    def dump_h(self, routines, f, prefix, header=True, empty=True):
        
        if header:
            print(''.join(self._get_header()), file=f)
        guard_name = "%s__%s__H" % (self.project.replace(
            " ", "_").upper(), prefix.replace("/", "_").upper())
        
        if empty:
            print(file=f)
        print("#ifndef %s" % guard_name, file=f)
        print("#define %s" % guard_name, file=f)
        if empty:
            print(file=f)
        
        print("#include <complex.h>", file=f)  

        if empty:
            print(file=f)
        for routine in routines:
            prototype = self.get_prototype(routine)
            print("%s;" % prototype, file=f)
        
        if empty:
            print(file=f)
        print("#endif", file=f)
        if empty:
            print(file=f)

    dump_h.extension = "h"
    dump_fns = [CCodeGen.dump_c, dump_h]

